# coding=utf-8
"""
Checkpoint 모델 평가 스크립트

학습된 checkpoint 모델을 불러와서 Seqorth MoE 분석을 수행합니다.
- Load balancing metrics
- Expert specialization metrics
- Routing quality metrics
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
import json
from tqdm import tqdm
import argparse
from peft import PeftModel
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoProcessor, AutoTokenizer
from transformers.generation.configuration_utils import GenerationConfig
from transformers.image_utils import load_image
import copy

from models import G3MoEModel, G3MoETextModel, G3MoEConfig, G3MoEForCausalLM, G3MoEForConditionalGeneration, G3MoETextConfig
from transformers.modeling_utils import VLMS
from evaluation.seqorth_analysis import SeqorthAnalyzer


# Register models
AutoConfig.register("g3moe", G3MoEConfig)
AutoConfig.register("g3moe_text", G3MoETextConfig)
AutoModel.register(G3MoEConfig, G3MoEModel)
AutoModel.register(G3MoETextConfig, G3MoETextModel)
AutoModelForCausalLM.register(G3MoEConfig, G3MoEForConditionalGeneration)
VLMS.append("g3moe")


class RoutingInfoCollector:
    """모델 forward pass에서 routing 정보를 수집하는 hook"""
    
    def __init__(self, analyzer: SeqorthAnalyzer):
        self.analyzer = analyzer
        self.hooks = []
        self.router_hooks = []
        self.routing_data = []
        self.router_internal_data = defaultdict(list)
        
    def register_hooks(self, model: nn.Module):
        """모델의 MoE 레이어와 Router에 hook 등록"""
        from models.g3moe_model import G3MoERouter, G3MoEGRINMoE
        from models.seqorth import SeqorthRouter, SeqorthBlock
        
        # Router의 forward hook (routing_logits, expression_logits 추출)
        def create_router_hook(layer_name):
            def router_hook_fn(module, input, output):
                # Router forward의 반환값: (multiplier, selected_experts, expression_logits, hn, speciality_penalty, cosine_similarities, expression_loss, routing_probs_full)
                # 8개 값을 반환 (routing_probs_full 추가됨)
                if len(output) >= 8:
                    multiplier, selected_experts, expression_logits, hn, speciality_penalty, cosine_similarities, expression_loss, routing_probs_full = output
                elif len(output) >= 7:
                    # 이전 버전 호환성 (7개만 반환하는 경우)
                    multiplier, selected_experts, expression_logits, hn, speciality_penalty, cosine_similarities, expression_loss = output
                    routing_probs_full = None
                else:
                    # output이 튜플이 아니거나 길이가 부족한 경우
                    return
                
                # Router 내부에서 routing_logits 추출 시도
                routing_logits = None
                if hasattr(module, 'load_balancer'):
                    # GRU의 출력을 routing_logits로 사용
                    # input[0]은 hidden_states, input[1]은 hn
                    if len(input) >= 1:
                        hidden_states = input[0]
                        hn_input = input[1] if len(input) > 1 else None
                        
                        # GRU forward를 직접 호출하여 routing_logits 얻기
                        with torch.no_grad():
                            if hn_input is not None:
                                routing_logits, _ = module.load_balancer(hidden_states, hn_input.to(hidden_states.dtype))
                            else:
                                routing_logits, _ = module.load_balancer(hidden_states, None)
                            
                            # Reshape to [batch, seq, num_experts, router_dim]
                            batch_size, seq_len = hidden_states.shape[:2]
                            num_experts = module.num_experts
                            router_dim = module.router_dim
                            routing_logits = routing_logits.view(batch_size, seq_len, num_experts, router_dim)
                
                # expression_logits의 shape 확인 및 수정
                # Router forward에서 expression_logits는 view(hidden_shape)를 거쳐 [batch, seq, num_experts, router_dim] 형태가 되어야 함
                # 하지만 실제로는 [batch, seq, 1, router_dim] 또는 [batch*seq, 1, router_dim] 형태일 수 있음
                expression_logits_fixed = expression_logits
                if isinstance(expression_logits, torch.Tensor):
                    # expression_logits의 shape 확인
                    if expression_logits.dim() == 4:
                        exp_batch, exp_seq, exp_num_exp, exp_router_dim = expression_logits.shape
                        if exp_num_exp == 1 and routing_logits is not None:
                            # [batch, seq, 1, router_dim] -> [batch, seq, num_experts, router_dim]로 expand
                            if routing_logits.dim() == 4:
                                _, _, num_experts, router_dim = routing_logits.shape
                                expression_logits_fixed = expression_logits.expand(exp_batch, exp_seq, num_experts, router_dim)
                    elif expression_logits.dim() == 3:
                        exp_batch_seq, exp_dim1, exp_dim2 = expression_logits.shape
                        if routing_logits is not None:
                            if routing_logits.dim() == 4:
                                batch_size, seq_len, num_experts, router_dim = routing_logits.shape
                                if exp_dim1 == 1 and exp_dim2 == router_dim:
                                    # [batch*seq, 1, router_dim] -> [batch*seq, num_experts, router_dim]로 expand
                                    expression_logits_fixed = expression_logits.expand(exp_batch_seq, num_experts, router_dim)
                                elif exp_dim1 * exp_dim2 == num_experts * router_dim:
                                    # [batch*seq, num_experts*router_dim] -> [batch*seq, num_experts, router_dim]로 reshape
                                    expression_logits_fixed = expression_logits.view(exp_batch_seq, num_experts, router_dim)
                            elif routing_logits.dim() == 3:
                                batch_seq_len, num_experts, router_dim = routing_logits.shape
                                if exp_dim1 == 1 and exp_dim2 == router_dim:
                                    expression_logits_fixed = expression_logits.expand(batch_seq_len, num_experts, router_dim)
                                elif exp_dim1 * exp_dim2 == num_experts * router_dim:
                                    expression_logits_fixed = expression_logits.view(batch_seq_len, num_experts, router_dim)
                
                self.router_internal_data[layer_name].append({
                    'routing_logits': routing_logits.detach().cpu() if routing_logits is not None else None,
                    'expression_logits': expression_logits_fixed.detach().cpu() if isinstance(expression_logits_fixed, torch.Tensor) else None,
                    'selected_experts': selected_experts.detach().cpu(),
                    'routing_weights': multiplier.detach().cpu(),
                    'cosine_similarities': cosine_similarities.detach().cpu() if isinstance(cosine_similarities, torch.Tensor) else None,
                    'speciality_penalty': float(speciality_penalty) if isinstance(speciality_penalty, torch.Tensor) else speciality_penalty,
                    'expression_loss': float(expression_loss) if isinstance(expression_loss, torch.Tensor) else expression_loss,
                })
            return router_hook_fn
        
        # MoE Block의 forward hook (G3MoEGRINMoE와 SeqorthBlock 모두 지원)
        def create_moe_hook(layer_name):
            def moe_hook_fn(module, input, output):
                # G3MoEGRINMoE: output = (final_hidden_states, (routing_weights, hn, speciality_loss, cosine_similarities, expression_loss))
                # 주의: forward에서 router_logits로 이름을 바꾸지만 실제로는 routing_weights입니다
                if isinstance(output, tuple) and len(output) == 2:
                    final_hidden_states, routing_info_tuple = output
                    if isinstance(routing_info_tuple, tuple) and len(routing_info_tuple) >= 5:
                        routing_weights_from_moe, hn, speciality_loss, cosine_similarities, expression_loss = routing_info_tuple[:5]
                        
                        # Router에서 수집한 데이터와 매칭
                        # Router hook이 먼저 실행되어 router_internal_data에 데이터가 저장되어야 함
                        # Global router를 사용하므로 모든 router 데이터 중 가장 최근 것을 사용
                        latest_router_data = None
                        if self.router_internal_data:
                            # 모든 router 데이터 중 가장 최근 것 찾기
                            all_router_data = []
                            for router_name, router_data_list in self.router_internal_data.items():
                                if router_data_list:
                                    all_router_data.extend([(router_name, data) for data in router_data_list])
                            
                            if all_router_data:
                                # 가장 최근 데이터 사용 (마지막 항목)
                                _, latest_router_data = all_router_data[-1]
                        
                        if latest_router_data:
                            
                            # routing_weights는 top-k에 대한 가중치이므로, router_scores를 재구성
                            # Router hook에서 수집한 selected_experts와 routing_weights를 사용
                            router_scores = None
                            selected_experts = latest_router_data.get('selected_experts')
                            routing_weights = latest_router_data.get('routing_weights')
                            
                            if selected_experts is not None and routing_weights is not None:
                                # selected_experts: [batch*seq, top_k]
                                # routing_weights: [batch*seq, top_k]
                                batch_size, seq_len = input[0].shape[:2] if len(input) > 0 and isinstance(input[0], torch.Tensor) else (1, 1)
                                num_experts = module.num_experts if hasattr(module, 'num_experts') else selected_experts.max().item() + 1
                                
                                # 모든 expert에 대한 점수를 0으로 초기화
                                router_scores = torch.zeros(batch_size * seq_len, num_experts, dtype=routing_weights.dtype)
                                
                                # selected_experts에 해당하는 위치에 routing_weights 할당
                                batch_seq_indices = torch.arange(batch_size * seq_len, device=selected_experts.device).unsqueeze(1).expand(-1, selected_experts.shape[-1])
                                router_scores[batch_seq_indices, selected_experts] = routing_weights
                                
                                router_scores = router_scores.view(batch_size, seq_len, num_experts)
                            
                            self.routing_data.append({
                                'layer': layer_name,
                                'routing_logits': latest_router_data.get('routing_logits'),
                                'expression_logits': latest_router_data.get('expression_logits'),
                                'routing_weights': latest_router_data.get('routing_weights'),
                                'selected_experts': latest_router_data.get('selected_experts'),
                                'cosine_similarities': latest_router_data.get('cosine_similarities'),
                                'speciality_penalty': latest_router_data.get('speciality_penalty', float(speciality_loss) if isinstance(speciality_loss, torch.Tensor) else speciality_loss),
                                'expression_loss': latest_router_data.get('expression_loss', float(expression_loss) if isinstance(expression_loss, torch.Tensor) else expression_loss),
                                'router_scores': router_scores.detach().cpu() if router_scores is not None else None,
                            })
                            return
                
                # SeqorthBlock: _last_routing_info 사용
                if hasattr(module, '_last_routing_info'):
                    routing_info = module._last_routing_info
                    if routing_info is not None and len(routing_info) >= 6:
                        routing_weights, hn, speciality_loss, cosine_similarities, expression_loss, router_scores = routing_info
                        
                        # Router에서 수집한 데이터와 매칭
                        # Global router를 사용하므로 모든 router 데이터 중 가장 최근 것을 사용
                        latest_router_data = None
                        if self.router_internal_data:
                            all_router_data = []
                            for router_name, router_data_list in self.router_internal_data.items():
                                if router_data_list:
                                    all_router_data.extend([(router_name, data) for data in router_data_list])
                            
                            if all_router_data:
                                _, latest_router_data = all_router_data[-1]
                        
                        if latest_router_data:
                            self.routing_data.append({
                                'layer': layer_name,
                                'routing_logits': latest_router_data.get('routing_logits'),
                                'expression_logits': latest_router_data.get('expression_logits'),
                                'routing_weights': latest_router_data.get('routing_weights'),
                                'selected_experts': latest_router_data.get('selected_experts'),
                                'cosine_similarities': latest_router_data.get('cosine_similarities'),
                                'speciality_penalty': latest_router_data.get('speciality_penalty', float(speciality_loss) if isinstance(speciality_loss, torch.Tensor) else speciality_loss),
                                'expression_loss': latest_router_data.get('expression_loss', float(expression_loss) if isinstance(expression_loss, torch.Tensor) else expression_loss),
                                'router_scores': router_scores.detach().cpu(),
                            })
            return moe_hook_fn
        
        # Router와 MoE Block에 hook 등록
        router_count = 0
        moe_count = 0
        
        for name, module in model.named_modules():
            # Router hook (G3MoERouter 또는 SeqorthRouter)
            if isinstance(module, (G3MoERouter, SeqorthRouter)) or (hasattr(module, 'load_balancer') and hasattr(module, 'expression_projector')):
                hook = module.register_forward_hook(create_router_hook(name))
                self.router_hooks.append(hook)
                router_count += 1
                print(f"Registered router hook: {name}")
            
            # MoE Block hook (G3MoEGRINMoE 또는 SeqorthBlock)
            if isinstance(module, (G3MoEGRINMoE, SeqorthBlock)) or hasattr(module, '_last_routing_info'):
                hook = module.register_forward_hook(create_moe_hook(name))
                self.hooks.append(hook)
                moe_count += 1
                print(f"Registered MoE block hook: {name}")
        
        print(f"\n✅ Hook registration complete: {router_count} routers, {moe_count} MoE blocks")
    
    def remove_hooks(self):
        """Hook 제거"""
        for hook in self.hooks + self.router_hooks:
            hook.remove()
        self.hooks = []
        self.router_hooks = []
    
    def analyze_collected_data(self, num_experts: int, router_dim: int = 128) -> Dict[str, Any]:
        """수집된 데이터 분석"""
        if not self.routing_data:
            print("⚠️  No routing data collected. Make sure hooks are registered correctly.")
            return {}
        
        all_metrics = []
        
        for data in self.routing_data:
            routing_logits = data.get('routing_logits')
            expression_logits = data.get('expression_logits')
            selected_experts = data.get('selected_experts')
            routing_weights = data.get('routing_weights')
            cosine_similarities = data.get('cosine_similarities')
            
            if routing_logits is None or expression_logits is None:
                # Router scores로부터 근사값 생성
                router_scores = data.get('router_scores')
                if router_scores is not None:
                    batch_size, seq_len, num_experts_actual = router_scores.shape
                    # Router scores를 기반으로 근사
                    routing_logits = router_scores.unsqueeze(-1).expand(-1, -1, -1, router_dim)
                    expression_logits = routing_logits.clone()
                else:
                    continue
            
            if selected_experts is None or routing_weights is None:
                # router_scores에서 추출
                router_scores = data.get('router_scores')
                if router_scores is not None:
                    batch_size, seq_len, num_experts_actual = router_scores.shape
                    top_k = routing_weights.shape[-1] if routing_weights is not None else 2
                    routing_scores_flat = router_scores.view(batch_size * seq_len, num_experts_actual)
                    top_k_values, selected_experts = torch.topk(routing_scores_flat, k=min(top_k, num_experts_actual), dim=-1)
                    routing_weights = torch.softmax(top_k_values, dim=-1)
                else:
                    continue
            
            # Shape 확인 및 변환
            # routing_logits와 expression_logits의 shape을 일치시킴
            if routing_logits is not None and expression_logits is not None:
                # routing_logits shape 확인
                batch_seq_len = None
                num_experts_actual = None
                router_dim_actual = None
                
                if routing_logits.dim() == 4:
                    batch_size, seq_len, num_experts_actual, router_dim_actual = routing_logits.shape
                    batch_seq_len = batch_size * seq_len
                    routing_logits = routing_logits.view(batch_seq_len, num_experts_actual, router_dim_actual)
                elif routing_logits.dim() == 3:
                    # [batch*seq, num_experts, router_dim] 형태
                    batch_seq_len, num_experts_actual, router_dim_actual = routing_logits.shape
                else:
                    print(f"⚠️ Unexpected routing_logits shape: {routing_logits.shape}")
                    continue
                
                # expression_logits shape 확인 및 변환
                if expression_logits.dim() == 4:
                    # [batch, seq, num_experts, router_dim]
                    exp_batch_size, exp_seq_len, exp_num_experts, exp_router_dim = expression_logits.shape
                    exp_batch_seq_len = exp_batch_size * exp_seq_len
                    if exp_batch_seq_len == batch_seq_len:
                        expression_logits = expression_logits.view(exp_batch_seq_len, exp_num_experts, exp_router_dim)
                    else:
                        # Shape이 다르면 재구성 시도
                        expression_logits = expression_logits.view(-1, exp_num_experts, exp_router_dim)
                elif expression_logits.dim() == 3:
                    # [batch*seq, num_experts, router_dim] 또는 [batch*seq, num_experts*router_dim]
                    exp_batch_seq_len, dim1, dim2 = expression_logits.shape
                    
                    if dim1 == num_experts_actual and dim2 == router_dim_actual:
                        # 이미 올바른 shape
                        pass
                    elif dim1 * dim2 == num_experts_actual * router_dim_actual:
                        # [batch*seq, num_experts*router_dim] 형태를 [batch*seq, num_experts, router_dim]로 변환
                        expression_logits = expression_logits.view(exp_batch_seq_len, num_experts_actual, router_dim_actual)
                    else:
                        # Shape이 맞지 않으면 재구성 시도
                        total_elements = expression_logits.numel()
                        expected_elements = batch_seq_len * num_experts_actual * router_dim_actual
                        
                        if total_elements == expected_elements:
                            expression_logits = expression_logits.view(batch_seq_len, num_experts_actual, router_dim_actual)
                        else:
                            print(f"⚠️ Cannot reshape expression_logits: shape={expression_logits.shape}, expected elements={expected_elements}, actual={total_elements}")
                            # 최선의 노력으로 재구성
                            if total_elements % (num_experts_actual * router_dim_actual) == 0:
                                new_batch_seq = total_elements // (num_experts_actual * router_dim_actual)
                                expression_logits = expression_logits.view(new_batch_seq, num_experts_actual, router_dim_actual)
                            else:
                                continue
                elif expression_logits.dim() == 2:
                    # [batch*seq, num_experts*router_dim] 형태
                    exp_batch_seq_len, exp_total_dim = expression_logits.shape
                    if exp_total_dim == num_experts_actual * router_dim_actual:
                        expression_logits = expression_logits.view(exp_batch_seq_len, num_experts_actual, router_dim_actual)
                    else:
                        print(f"⚠️ Cannot reshape expression_logits from 2D: shape={expression_logits.shape}, expected dim={num_experts_actual * router_dim_actual}")
                        continue
                else:
                    print(f"⚠️ Unexpected expression_logits shape: {expression_logits.shape}")
                    continue
                
                # 최종 shape 확인
                if routing_logits.shape != expression_logits.shape:
                    print(f"⚠️ Shape mismatch after conversion: routing_logits={routing_logits.shape}, expression_logits={expression_logits.shape}")
                    # 최소한의 shape으로 맞춤
                    min_batch_seq = min(routing_logits.shape[0], expression_logits.shape[0])
                    routing_logits = routing_logits[:min_batch_seq]
                    expression_logits = expression_logits[:min_batch_seq]
            
            if cosine_similarities is None:
                batch_size, seq_len = routing_logits.shape[0] // num_experts, 1
                cosine_similarities = torch.zeros(batch_size, seq_len, num_experts)
            
            # Analyzer에 전달
            try:
                metrics = self.analyzer.analyze_routing_step(
                    routing_logits=routing_logits,
                    expression_logits=expression_logits,
                    selected_experts=selected_experts,
                    routing_weights=routing_weights,
                    speciality_penalty=data.get('speciality_penalty', 0.0),
                    cosine_similarities=cosine_similarities,
                    expression_loss=data.get('expression_loss', 0.0),
                )
                
                metrics['layer'] = data['layer']
                all_metrics.append(metrics)
            except Exception as e:
                print(f"Error analyzing data for layer {data.get('layer', 'unknown')}: {e}")
                continue
        
        return {
            'per_layer_metrics': all_metrics,
            'aggregated_metrics': self.analyzer.get_aggregated_metrics(),
            'paper_summary': self.analyzer.get_paper_metrics_summary(),
        }


def load_checkpoint_model(
    checkpoint_path: str,
    base_model_name: str,
    model_architecture,
    moe_config: Dict[str, Any],
    device: str = "cuda",
) -> Tuple[nn.Module, Any]:
    """Checkpoint에서 모델과 토크나이저 로드"""
    print(f"Loading base model: {base_model_name}")
    base_config = AutoConfig.from_pretrained(base_model_name, trust_remote_code=True)
    base_config = base_config.to_dict()
    
    if "text_config" not in base_config:
        base_config['text_config'] = copy.deepcopy(base_config)
    
    base_config['text_config'].update(moe_config)
    base_config.update(base_config['text_config'])
    model_config = G3MoEConfig(**base_config)
    model_config.model_type = "gemma3"
    model_config.text_config.model_type = "gemma3_text"
    model_config.architectures = ["G3MoEForConditionalGeneration"]
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    model = PeftModel.from_pretrained(
        model=model_architecture.from_pretrained(
            pretrained_model_name_or_path=base_model_name,
            config=model_config,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            offload_state_dict=True,
            use_cache=False,
            attn_implementation="flash_attention_3",
        ).to(device),
        model_id=checkpoint_path,
    )
    model.merge_and_unload()
    model.eval()
    
    # Tokenizer 로드
    try:
        tokenizer = AutoProcessor.from_pretrained(base_model_name, use_fast=True)
        if hasattr(tokenizer, 'chat_template'):
            chat_template_path = "/home/conan/workspace/llm_training/sft/config/chat_template.txt"
            if os.path.exists(chat_template_path):
                with open(chat_template_path, "r") as f:
                    tokenizer.chat_template = f.read()
    except:
        tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)
    
    print("✅ Model and tokenizer loaded successfully")
    return model, tokenizer


def prepare_evaluation_data(
    tokenizer: Any,
    dataset_name: Optional[str] = None,
    num_samples: int = 500,
    max_length: int = 512,
    use_training_eval_set: bool = True,
) -> List[Dict[str, torch.Tensor]]:
    """
    평가용 데이터 준비 (streaming 모드)
    
    HuggingFace Hub에서 streaming 모드로 데이터를 로드합니다.
    """
    from datasets import load_dataset
    from datasets.iterable_dataset import IterableDataset
    from itertools import islice
    
    inputs_list = []
    
    if not dataset_name:
        raise ValueError("dataset_name must be provided")
    
    # Option 1: 학습에 사용한 test split 사용
    if use_training_eval_set:
        from data.simple_sft_dataset import get_simple_sft_dataset
        print(f"Loading training eval set from: {dataset_name} (streaming mode)")
        
        dataset = get_simple_sft_dataset(
            dataset_name=dataset_name,
            tokenizer=tokenizer,
            max_length=max_length,
            max_samples=num_samples,
            test_size=0.1,
            use_streaming=True
        )
        
        eval_dataset = dataset.get("test", None)
        if eval_dataset is None:
            raise ValueError(f"No test split found in dataset: {dataset_name}")
        
        print(f"✅ Loaded eval dataset (streaming mode)")
        
        # Streaming 데이터셋 처리
        sample_count = 0
        for sample in tqdm(eval_dataset, desc="Preparing eval data", total=num_samples):
            if sample_count >= num_samples:
                break
            
            # VLM 데이터셋인 경우 이미지 포함
            if 'images' in sample and sample['images']:
                messages = sample.get('messages', [])
                if not messages:
                    continue
                
                # Chat template 적용
                if hasattr(tokenizer, 'apply_chat_template'):
                    text = tokenizer.apply_chat_template(
                        messages,
                        add_generation_prompt=True,
                        tokenize=False
                    )
                else:
                    text = str(messages)
                
                # 이미지와 텍스트 함께 처리
                images = sample['images']
                if isinstance(images, list) and len(images) > 0:
                    inputs = tokenizer(
                        text=text,
                        images=images[0] if len(images) == 1 else images,
                        return_tensors="pt",
                        truncation=True,
                        max_length=max_length,
                        padding="max_length",
                    )
                else:
                    inputs = tokenizer(
                        text=text,
                        return_tensors="pt",
                        truncation=True,
                        max_length=max_length,
                        padding="max_length",
                    )
            else:
                # 텍스트만 있는 경우
                messages = sample.get('messages', [])
                if not messages:
                    continue
                
                if hasattr(tokenizer, 'apply_chat_template'):
                    text = tokenizer.apply_chat_template(
                        messages,
                        add_generation_prompt=True,
                        tokenize=False
                    )
                else:
                    text = str(messages)
                
                inputs = tokenizer(
                    text=text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_length,
                    padding="max_length",
                )
            
            if "token_type_ids" in inputs:
                del inputs["token_type_ids"]
            inputs_list.append(inputs)
            sample_count += 1
        
        print(f"✅ Successfully loaded {len(inputs_list)} samples from {dataset_name}")
        return inputs_list
    
    # Option 2: HuggingFace Hub에서 직접 로드 (streaming 모드)
    print(f"Loading dataset from HuggingFace Hub: {dataset_name} (streaming mode)")
    
    # 데이터셋 이름에서 split 추출 (형식: "dataset_name:split" 필수)
    if ':' not in dataset_name:
        raise ValueError(
            f"Dataset name must include split in format 'dataset_name:split'. "
            f"Got: {dataset_name}. Example: 'lmms-lab/VQAv2:validation'"
        )
    
    dataset_path, split_name = dataset_name.split(':', 1)
    
    try:
        # Streaming 모드로 데이터셋 로드
        dataset = load_dataset(
            dataset_path,
            split=split_name,
            streaming=True
        )
        
        if not isinstance(dataset, IterableDataset):
            raise ValueError(f"Expected IterableDataset, got {type(dataset)}")
        
        print(f"✅ Loaded dataset in streaming mode")
        
        # Streaming 데이터셋에서 샘플 추출
        sample_count = 0
        for sample in tqdm(islice(dataset, num_samples), desc="Preparing eval data", total=num_samples):
            try:
                # 이미지 처리
                image = None
                if 'image' in sample:
                    image = sample['image']
                elif 'images' in sample:
                    images = sample['images']
                    if isinstance(images, list) and len(images) > 0:
                        image = images[0]
                    elif images:
                        image = images
                
                # 텍스트/질문 처리
                question = None
                if 'question' in sample:
                    question = sample['question']
                elif 'text' in sample:
                    question = sample['text']
                elif 'prompt' in sample:
                    question = sample['prompt']
                elif 'messages' in sample:
                    messages = sample['messages']
                    if hasattr(tokenizer, 'apply_chat_template'):
                        question = tokenizer.apply_chat_template(
                            messages,
                            add_generation_prompt=True,
                            tokenize=False
                        )
                    else:
                        question = str(messages)
                
                if not question:
                    continue
                
                # 프롬프트 생성
                if image is not None:
                    prompt =  tokenizer.apply_chat_template(
                            [
                                {
                                    "role": "system",
                                    "content": [
                                        {"type": "text", "text": "You are a helpful assistant."}
                                    ]
                                },
                                {
                                    "role": "user",
                                    "content": [
                                        {"type": "text", "text": question},
                                        {"type": "image"}
                                    ]
                                }
                            ],
                            # tokenize=True,
                            add_generation_prompt=True,
                            # return_tensors="pt",
                            # return_dict=True,
                        )
                    inputs = tokenizer(
                        text=prompt,
                        images=[image],
                        return_tensors="pt",
                        truncation=True,
                        max_length=max_length,
                        padding="max_length",
                    )
                else:
                    prompt =  tokenizer.apply_chat_template(
                            [
                                {
                                    "role": "system",
                                    "content": [
                                        {"type": "text", "text": "You are a helpful assistant."}
                                    ]
                                },
                                {
                                    "role": "user",
                                    "content": [
                                        {"type": "text", "text": question}
                                    ]
                                }
                            ],
                            # tokenize=True,
                            add_generation_prompt=True,
                            # return_tensors="pt",
                            # return_dict=True,
                        )
                    inputs = tokenizer(
                        text=prompt,
                        return_tensors="pt",
                        truncation=True,
                        max_length=max_length,
                        padding="max_length",
                    )
                
                if "token_type_ids" in inputs:
                    del inputs["token_type_ids"]
                
                # 레이블 정보가 있으면 함께 저장
                if 'label' in sample:
                    inputs['label'] = sample['label']
                if 'label_text' in sample:
                    inputs['label_text'] = sample['label_text']
                if 'answer' in sample:
                    inputs['answer'] = sample['answer']
                
                inputs_list.append(inputs)
                sample_count += 1
                
            except Exception as e:
                print(f"⚠️ Error processing sample: {e}")
                continue
        
        print(f"✅ Successfully loaded {len(inputs_list)} samples from {dataset_name}")
        return inputs_list
        
    except Exception as e:
        import traceback
        print(f"❌ Failed to load dataset {dataset_name}: {e}")
        traceback.print_exc()
        raise


def evaluate_model(
    model: nn.Module,
    tokenizer: Any,
    eval_data: List[Dict[str, torch.Tensor]],
    analyzer: SeqorthAnalyzer,
    device: str = "cuda",
    max_samples: Optional[int] = None,
) -> Dict[str, Any]:
    """모델 평가 실행"""
    print(f"\n{'='*60}")
    print("Starting Model Evaluation")
    print(f"{'='*60}")
    
    collector = RoutingInfoCollector(analyzer)
    collector.register_hooks(model)
    
    model.eval()
    
    # max_samples가 지정된 경우 제한
    if max_samples:
        eval_data = eval_data[:max_samples]
    
    num_samples = len(eval_data)
    print(f"Evaluating on {num_samples} samples...")
    
    with torch.no_grad():
        for i, inputs in enumerate(tqdm(eval_data, desc="Evaluating", total=num_samples)):
            # Move to device
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            
            # Forward pass
            try:
                outputs = model(**inputs)
            except Exception as e:
                print(f"Error in forward pass for sample {i}: {e}")
                continue
    
    # 분석 수행
    print("\nAnalyzing collected routing data...")
    results = collector.analyze_collected_data(
        num_experts=analyzer.num_experts,
        router_dim=analyzer.router_dim,
    )
    
    collector.remove_hooks()
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate checkpoint model with Seqorth analysis")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to checkpoint directory")
    parser.add_argument(
        "--base_model",
        type=str,
        default="Gunulhona/Gemma-3-4B",
        help="Base model name")
    parser.add_argument(
        "--num_samples",
        type=int,
        default=100,
        help="Number of evaluation samples")
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum sequence length")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./evaluation_results",
        help="Output directory for results")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use")
    parser.add_argument(
        "--eval_dataset",
        type=str,
        required=True,
        help="Evaluation dataset name from HuggingFace Hub. Format: 'dataset_name:split' (e.g., 'lmms-lab/VQAv2:validation'). All datasets are loaded in streaming mode.")
    parser.add_argument(
        "--use_training_eval_set",
        action="store_true",
        help="Use test split from training dataset (streaming mode)")
    
    args = parser.parse_args()
    
    # MoE config (checkpoint에서 가져와야 하지만, 여기서는 기본값 사용)
    moe_config = {
        "n_shared_experts": 1,
        "n_routed_experts": 8,
        "n_group": 2,
        "topk_group": 2,
        "num_experts_per_tok": 2,
        "first_k_dense_replace": 8,
        "router_aux_loss_coef": 9e-1,
        "router_jitter_noise": 1e-05,
        "input_jitter_noise": 1e-05,
        "router_z_loss_coef": 1e-2,
        "ema_alpha": 0.99,
        "balancing_strength": 5e-2,
        "no_rope_layer_interval": 4,
        "use_sliding_window": True,
        "rope_scaling": {
            "rope_type": "yarn",
            "factor": 8.0
        },
        "use_bfloat16": True
    }
    
    # 모델 로드
    model, tokenizer = load_checkpoint_model(
        checkpoint_path=args.checkpoint_path,
        base_model_name=args.base_model,
        model_architecture=G3MoEForConditionalGeneration,
        moe_config=moe_config,
        device=args.device,
    )
    
    # Analyzer 초기화
    analyzer = SeqorthAnalyzer(
        num_experts=moe_config.get('n_routed_experts', 8),
        router_dim=moe_config.get('router_dim', 128),
    )
    
    # 평가 데이터 준비
    eval_data = prepare_evaluation_data(
        tokenizer=tokenizer,
        dataset_name=args.eval_dataset,
        num_samples=args.num_samples,
        max_length=args.max_length,
        use_training_eval_set=args.use_training_eval_set,
    )
    
    # 평가 실행
    results = evaluate_model(
        model=model,
        tokenizer=tokenizer,
        eval_data=eval_data,
        analyzer=analyzer,
        device=args.device,
    )
    
    # 결과 저장
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, "evaluation_results.json")
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("Evaluation Complete!")
    print(f"{'='*60}")
    print(f"Results saved to: {output_file}")
    
    # 논문용 요약 출력
    if 'paper_summary' in results:
        print("\n📊 Paper Summary Metrics:")
        print(json.dumps(results['paper_summary'], indent=2))
    
    # 집계 지표 출력
    if 'aggregated_metrics' in results:
        print("\n📈 Aggregated Metrics:")
        agg = results['aggregated_metrics']
        
        def format_metric(value, default='N/A'):
            if value == default or value is None:
                return default
            try:
                return f"{float(value):.4f}"
            except (ValueError, TypeError):
                return str(value)
        
        print("\n📊 주요 Load Balancing 지표:")
        print(f"  Load Balancing CV: {format_metric(agg.get('final_load_balancing_cv', 'N/A'))}")
        print(f"  Load Imbalance Ratio: {format_metric(agg.get('final_load_imbalance_ratio', 'N/A'))}")
        print(f"  MaxVio (Maximum Violation): {format_metric(agg.get('final_maxvio', 'N/A'))}")
        print(f"  Aux Loss: {format_metric(agg.get('final_aux_loss', 'N/A'))}")
        print(f"  Expert Utilization Rate: {format_metric(agg.get('expert_utilization_rate', 'N/A'))}")
        
        print("\n📈 최근 논문 지표:")
        print(f"  LPR (Layer-wise Performance Ratio): {format_metric(agg.get('final_lpr', 'N/A'))}")
        print(f"  Expert Efficiency (DeepSpeed MoE): {format_metric(agg.get('final_expert_efficiency', 'N/A'))}")
        print(f"  Expert Capacity Utilization: {format_metric(agg.get('avg_expert_capacity_utilization', 'N/A'))}")
        print(f"  Load Variance: {format_metric(agg.get('avg_load_variance', 'N/A'))}")
        
        print("\n🔬 Gram Matrix & Specialization 지표:")
        print(f"  Gram Orthogonality (평균): {format_metric(agg.get('avg_gram_orthogonality', 'N/A'))}")
        if 'std_gram_orthogonality' in agg:
            print(f"  Gram Orthogonality (표준편차): {format_metric(agg.get('std_gram_orthogonality', 'N/A'))}")


if __name__ == "__main__":
    main()

