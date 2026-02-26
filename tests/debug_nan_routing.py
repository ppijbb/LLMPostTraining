#!/usr/bin/env python3
"""
NaN 라우팅 가중치 디버깅 스크립트
각 레이어의 출력에서 처음으로 NaN이 발생하는 경우를 찾습니다.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Union, List
import math
import os
import sys
import traceback

# 현재 디렉토리를 경로에 추가
sys.path.insert(0, '/home/conan/workspace/llm_training')

# 필요한 모듈 임포트
from models.seqorth_model import SeqorthForCausalLM, SeqorthForConditionalGeneration, SeqorthModel, SeqorthTextModel, SeqorthConfig, SeqorthTextConfig
from transformers import AutoTokenizer, AutoConfig, AutoModel, AutoModelForCausalLM
from transformers.modeling_utils import VLMS

try:
    # AutoConfig.register("seqorth", SeqorthConfig)
    AutoConfig.register("seqorth", SeqorthConfig)
    AutoConfig.register("seqorth_text", SeqorthTextConfig)
    AutoModel.register(SeqorthConfig, SeqorthModel)
    AutoModel.register(SeqorthTextConfig, SeqorthTextModel)
    AutoModelForCausalLM.register(SeqorthConfig, SeqorthForConditionalGeneration)
    VLMS.append("seqorth")
except Exception as e:
    traceback.format_exc()
    print(f"Failed to register Seqorth model: {e}")
    print("Seqorth cannot train without registering model... exiting...")
    raise e

def create_simple_causal_mask(
    batch_size: int,
    seq_length: int,
    dtype: torch.dtype,
    device: torch.device,
    attention_mask_2d: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    단순한 4D causal attention mask를 생성합니다.
    
    Args:
        batch_size: 배치 크기
        seq_length: 시퀀스 길이  
        dtype: 텐서의 dtype
        device: 텐서의 device
        attention_mask_2d: 선택적 2D padding mask [batch_size, seq_length]
    
    Returns:
        4D causal mask [batch_size, 1, seq_length, seq_length]
    """
    # Lower triangular causal mask 생성
    # 1 = attend, 0 = mask
    causal_mask = torch.tril(torch.ones((seq_length, seq_length), dtype=dtype, device=device))
    
    # 4D로 확장 [1, 1, seq_len, seq_len]
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
    
    # 배치 차원으로 확장 [batch_size, 1, seq_len, seq_len]
    causal_mask = causal_mask.expand(batch_size, 1, seq_length, seq_length).clone()
    
    # attention에서는 0 = attend, min_dtype = mask 형태로 사용
    # NOTE: -inf 대신 dtype의 min 값 사용 (softmax에서 NaN 방지)
    min_dtype = torch.finfo(dtype).min
    causal_mask = torch.where(
        causal_mask.bool(),
        torch.zeros_like(causal_mask),
        torch.full_like(causal_mask, min_dtype)
    )
    
    # 2D padding mask가 있으면 결합
    if attention_mask_2d is not None:
        # attention_mask_2d: [batch_size, seq_length] -> [batch_size, 1, 1, seq_length]
        padding_mask = attention_mask_2d.unsqueeze(1).unsqueeze(2)
        padding_mask = padding_mask.to(dtype=dtype)
        # 0인 위치(padding)에 min_dtype 적용
        padding_mask = torch.where(
            padding_mask.bool(),
            torch.zeros_like(padding_mask),
            torch.full_like(padding_mask, min_dtype)
        )
        # 브로드캐스팅으로 결합 - clamping으로 오버플로우 방지
        causal_mask = torch.clamp(causal_mask + padding_mask, min=min_dtype)
    
    return causal_mask


def debug_forward_with_nan_detection(
    model: SeqorthForCausalLM,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    max_layers_to_check: Optional[int] = None,
    verbose: bool = True
) -> Tuple[Optional[int], Optional[str], Optional[torch.Tensor]]:
    """
    모델의 forward 패스를 디버깅하여 NaN이 발생하는 레이어를 찾습니다.

    Args:
        model: 디버깅할 Seqorth 모델
        input_ids: 입력 토큰 ID
        attention_mask: 어텐션 마스크
        max_layers_to_check: 디버깅할 최대 레이어 수 (None이면 모든 레이어)
        verbose: 디버그 정보 출력 여부

    Returns:
        tuple: (nan_layer_idx, nan_component, nan_tensor)
            nan_layer_idx: NaN이 발생한 레이어 인덱스 (None이면 발생하지 않음)
            nan_component: NaN이 발생한 컴포넌트 이름
            nan_tensor: NaN이 포함된 텐서
    """

    # 모델을 평가 모드로 설정
    model.eval()

    # 입력 임베딩 얻기
    with torch.no_grad():
        inputs_embeds = model.get_input_embeddings()(input_ids)

    # 초기 상태 설정
    hidden_states = inputs_embeds
    past_key_values = None
    cache_position = None
    global_routing_hn = None

    # 모델 구성 가져오기
    config = model.config
    num_layers = len(model.model.layers)
    
    # position_ids 생성
    batch_size, seq_length = input_ids.shape
    position_ids = torch.arange(seq_length, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)

    # 4D causal attention mask 생성 (중요!)
    causal_mask_4d = create_simple_causal_mask(
        batch_size=batch_size,
        seq_length=seq_length,
        dtype=hidden_states.dtype,
        device=hidden_states.device,
        attention_mask_2d=attention_mask
    )
    
    if verbose:
        print(f"4D Causal mask 생성 완료: shape={causal_mask_4d.shape}")

    # 디버깅할 레이어 수 결정
    layers_to_check = min(num_layers, max_layers_to_check) if max_layers_to_check is not None else num_layers

    if verbose:
        print(f"디버깅 시작: 총 {num_layers} 레이어 중 {layers_to_check} 레이어 검사")
        print(f"입력 shape: {input_ids.shape}")
        print(f"입력 임베딩 shape: {inputs_embeds.shape}")

    # 레이어별로 순차적으로 실행
    for layer_idx in range(layers_to_check):
        if verbose:
            print(f"\n=== 레이어 {layer_idx} 디버깅 ===")

        # 현재 레이어 가져오기
        decoder_layer = model.model.layers[layer_idx]

        # 레이어 입력 검증
        if torch.isnan(hidden_states).any():
            return layer_idx, "hidden_states_input", hidden_states

        # 레이어 실행 - 각 sub-component별로 분리하여 NaN 추적
        try:
            with torch.no_grad():
                # 1. Input LayerNorm
                residual = hidden_states
                normed_hidden = decoder_layer.input_layernorm(hidden_states)
                if verbose:
                    nan_count = torch.isnan(normed_hidden).sum().item()
                    print(f"  [1] input_layernorm 후: NaN count = {nan_count}, mean = {normed_hidden.mean().item():.6f}")
                    if nan_count > 0:
                        return layer_idx, "input_layernorm", normed_hidden
                
                # 2. Self Attention - 수동 디버깅
                self_attn = decoder_layer.self_attn
                
                # 2.1 Q, K, V projection
                input_shape = normed_hidden.shape[:-1]
                hidden_shape = (*input_shape, -1, self_attn.head_dim)
                
                query_states = self_attn.q_proj(normed_hidden).view(hidden_shape).transpose(1, 2)
                if verbose:
                    nan_count = torch.isnan(query_states).sum().item()
                    print(f"    [2.1] q_proj 후: NaN count = {nan_count}")
                    if nan_count > 0:
                        return layer_idx, "q_proj", query_states
                
                key_states = self_attn.k_proj(normed_hidden).view(hidden_shape).transpose(1, 2)
                if verbose:
                    nan_count = torch.isnan(key_states).sum().item()
                    print(f"    [2.2] k_proj 후: NaN count = {nan_count}")
                    if nan_count > 0:
                        return layer_idx, "k_proj", key_states
                
                value_states = self_attn.v_proj(normed_hidden).view(hidden_shape).transpose(1, 2)
                if verbose:
                    nan_count = torch.isnan(value_states).sum().item()
                    print(f"    [2.3] v_proj 후: NaN count = {nan_count}")
                    if nan_count > 0:
                        return layer_idx, "v_proj", value_states
                
                # 2.2 Q, K normalization
                query_states = self_attn.q_norm(query_states)
                key_states = self_attn.k_norm(key_states)
                if verbose:
                    nan_count_q = torch.isnan(query_states).sum().item()
                    nan_count_k = torch.isnan(key_states).sum().item()
                    print(f"    [2.4] q_norm/k_norm 후: Q NaN = {nan_count_q}, K NaN = {nan_count_k}")
                    if nan_count_q > 0:
                        return layer_idx, "q_norm", query_states
                    if nan_count_k > 0:
                        return layer_idx, "k_norm", key_states
                
                # 2.3 repeat_kv for multi-head
                n_rep = self_attn.num_key_value_groups
                if n_rep > 1:
                    batch, num_kv_heads, slen, head_dim = key_states.shape
                    key_states_rep = key_states[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, slen, head_dim)
                    key_states_rep = key_states_rep.reshape(batch, num_kv_heads * n_rep, slen, head_dim)
                    value_states_rep = value_states[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, slen, head_dim)
                    value_states_rep = value_states_rep.reshape(batch, num_kv_heads * n_rep, slen, head_dim)
                else:
                    key_states_rep = key_states
                    value_states_rep = value_states
                
                # 2.4 Attention weights
                scaling = self_attn.scaling
                attn_weights = torch.matmul(query_states, key_states_rep.transpose(2, 3)) * scaling
                if verbose:
                    nan_count = torch.isnan(attn_weights).sum().item()
                    print(f"    [2.5] Q@K^T 후: NaN count = {nan_count}, min = {attn_weights.min().item():.4f}, max = {attn_weights.max().item():.4f}")
                    if nan_count > 0:
                        return layer_idx, "qk_matmul", attn_weights
                
                # 2.5 Apply causal mask
                if causal_mask_4d is not None:
                    causal_mask = causal_mask_4d[:, :, :, : key_states_rep.shape[-2]]
                    attn_weights = attn_weights + causal_mask
                    if verbose:
                        nan_count = torch.isnan(attn_weights).sum().item()
                        inf_count = torch.isinf(attn_weights).sum().item()
                        print(f"    [2.6] +causal_mask 후: NaN = {nan_count}, Inf = {inf_count}")
                        # Check for all-masked rows (all -inf)
                        all_masked_rows = (attn_weights == float('-inf')).all(dim=-1).sum().item()
                        print(f"    [2.6.1] 모든 토큰이 masked된 row 수: {all_masked_rows}")
                
                # 2.6 Softmax
                attn_weights_soft = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
                if verbose:
                    nan_count = torch.isnan(attn_weights_soft).sum().item()
                    print(f"    [2.7] softmax 후: NaN count = {nan_count}")
                    if nan_count > 0:
                        print(f"    ***** SOFTMAX에서 NaN 발생! *****")
                        print(f"    attn_weights 중 -inf 비율: {(attn_weights == float('-inf')).float().mean().item():.4f}")
                        return layer_idx, "softmax", attn_weights_soft
                
                # 2.7 Matmul with values
                attn_output = torch.matmul(attn_weights_soft, value_states_rep)
                attn_output = attn_output.transpose(1, 2).contiguous()
                if verbose:
                    nan_count = torch.isnan(attn_output).sum().item()
                    print(f"    [2.8] attn@V 후: NaN count = {nan_count}")
                    if nan_count > 0:
                        return layer_idx, "attn_value_matmul", attn_output
                
                # 2.8 Output projection
                attn_output = attn_output.reshape(*input_shape, -1).contiguous()
                attn_output = self_attn.o_proj(attn_output)
                if verbose:
                    nan_count = torch.isnan(attn_output).sum().item()
                    print(f"  [2] self_attn 후 (최종): NaN count = {nan_count}")
                    if nan_count > 0:
                        return layer_idx, "o_proj", attn_output
                
                # 3. Post Attention LayerNorm
                attn_output = decoder_layer.post_attention_layernorm(attn_output)
                if verbose:
                    nan_count = torch.isnan(attn_output).sum().item()
                    print(f"  [3] post_attention_layernorm 후: NaN count = {nan_count}")
                    if nan_count > 0:
                        return layer_idx, "post_attention_layernorm", attn_output
                
                # 4. residual 연결
                hidden_states = residual + attn_output
                if verbose:
                    nan_count = torch.isnan(hidden_states).sum().item()
                    print(f"  [4] residual + attn_output 후: NaN count = {nan_count}")
                    if nan_count > 0:
                        return layer_idx, "attn_residual", hidden_states
                
                # 5. Pre Feedforward LayerNorm
                residual = hidden_states
                normed_for_ffn = decoder_layer.pre_feedforward_layernorm(hidden_states)
                if verbose:
                    nan_count = torch.isnan(normed_for_ffn).sum().item()
                    print(f"  [5] pre_feedforward_layernorm 후: NaN count = {nan_count}")
                    if nan_count > 0:
                        return layer_idx, "pre_feedforward_layernorm", normed_for_ffn
                
                # 6. MoE / MLP
                if hasattr(decoder_layer.moe, 'experts'):
                    # MoE 레이어
                    moe_output, routing_info = decoder_layer.moe(normed_for_ffn, global_routing_hn)
                    global_routing_hn = routing_info[1] if routing_info[1] is not None else global_routing_hn
                else:
                    # Dense MLP 레이어
                    moe_output = decoder_layer.moe(normed_for_ffn)
                    routing_info = (None,) * 12
                
                if verbose:
                    nan_count = torch.isnan(moe_output).sum().item()
                    print(f"  [6] MoE/MLP 후: NaN count = {nan_count}")
                    if nan_count > 0:
                        return layer_idx, "moe_output", moe_output
                
                # 7. Post Feedforward LayerNorm
                moe_output = decoder_layer.post_feedforward_layernorm(moe_output)
                if verbose:
                    nan_count = torch.isnan(moe_output).sum().item()
                    print(f"  [7] post_feedforward_layernorm 후: NaN count = {nan_count}")
                    if nan_count > 0:
                        return layer_idx, "post_feedforward_layernorm", moe_output
                
                # 8. FFN residual 연결
                hidden_states = residual + moe_output
                if verbose:
                    nan_count = torch.isnan(hidden_states).sum().item()
                    print(f"  [8] residual + moe_output 후: NaN count = {nan_count}")
                    if nan_count > 0:
                        return layer_idx, "ffn_residual", hidden_states
                
                # layer_outputs 형식으로 재구성 (호환성)
                layer_outputs = (hidden_states, None, routing_info)

            # 레이어 출력 검증
            if len(layer_outputs) < 2:
                if verbose:
                    print(f"레이어 {layer_idx} 출력 형식이 잘못됨: {len(layer_outputs)} 개 출력")
                continue

            # hidden_states 검증
            new_hidden_states = layer_outputs[0]
            if torch.isnan(new_hidden_states).any():
                return layer_idx, "hidden_states_output", new_hidden_states

            # 라우팅 정보 검증 (MoE 레이어인 경우)
            routing_result = layer_outputs[-1]
            if routing_result is not None and len(routing_result) > 0:
                # 라우팅 정보 검증
                for i, item in enumerate(routing_result):
                    if item is not None and torch.is_tensor(item):
                        if torch.isnan(item).any():
                            component_names = [
                                "router_logits", "hn", "speciality_loss", "cosine_similarities",
                                "contrastive_loss", "expression_reg_loss", "routing_uncertainty",
                                "entropy_loss", "load_balancing_loss", "sinkhorn_loss",
                                "ortho_loss", "balance_loss"
                            ]
                            component_name = component_names[i] if i < len(component_names) else f"routing_item_{i}"
                            return layer_idx, f"routing_{component_name}", item

                # routing_weights 검증 (MoE 레이어인 경우)
                if hasattr(decoder_layer.moe, 'last_routing_weights') and decoder_layer.moe.last_routing_weights is not None:
                    routing_weights = decoder_layer.moe.last_routing_weights
                    if torch.isnan(routing_weights).any():
                        return layer_idx, "routing_weights", routing_weights

            # 상태 업데이트
            hidden_states = new_hidden_states

            if verbose:
                traceback.print_exc()
                print(f"레이어 {layer_idx} 통과: hidden_states shape={hidden_states.shape}")

        except Exception as e:
            if verbose:
                traceback.print_exc()
                print(f"레이어 {layer_idx} 실행 중 오류 발생: {str(e)}")
            return layer_idx, f"layer_exception_{str(e)}", None

    if verbose:
        print(f"\n디버깅 완료: {layers_to_check} 레이어 검사 완료, NaN 미발견")

    return None, None, None

def create_debug_model_and_inputs(
    model_name: str = "Qwen/Qwen3-VL-30B-A3B-Instruct",
    sequence_length: int = 8,  # 16 -> 8으로 더 감소
    batch_size: int = 1,
    device: str = "cpu"  # CUDA 메모리 캐시 문제로 인해 CPU로 강제 설정
) -> Tuple[SeqorthForCausalLM, torch.Tensor, torch.Tensor]:
    """
    디버깅용 모델과 입력 데이터를 생성합니다.

    Args:
        model_name: 모델 이름
        sequence_length: 시퀀스 길이
        batch_size: 배치 크기
        device: 디바이스

    Returns:
        tuple: (model, input_ids, attention_mask)
    """

    print(f"모델 로딩: {model_name}")
    print(f"디바이스: {device} (CUDA 메모리 캐시 문제로 인해 CPU 사용)")

    # 간단한 구성 생성
    base_config = AutoConfig.from_pretrained(model_name)
    config = SeqorthConfig(
        text_config=SeqorthTextConfig(
            **base_config.text_config.to_dict()
        ),
        vision_config=base_config.vision_config
    )
    
    # 최소한의 구성으로 설정
    config.model_type = "seqorth"
    config.text_config.model_type = "seqorth_text"
    config.text_config.hidden_size = 64
    config.text_config.intermediate_size = 256
    config.text_config.num_hidden_layers = 2
    config.text_config.num_attention_heads = 2
    config.text_config.num_key_value_heads = 1
    config.text_config.n_routed_experts = 2
    config.text_config.num_experts_per_tok = 1
    config.text_config.router_dim = 8
    config.text_config.sliding_window_pattern = 6
    config.text_config.first_k_dense_replace = 0
    config.text_config._attn_implementation = "eager"
    # 모델 생성 (CPU에서 실행)
    device = "cuda"
    model = SeqorthForCausalLM(config)
    model = model.to(device)
    model.eval()

    # 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 더미 입력 생성
    dummy_text = "This is a test sentence for debugging NaN issues in routing weights."
    inputs = tokenizer(
        dummy_text,
        return_tensors="pt",
        padding="max_length",
        max_length=sequence_length,
        truncation=True
    )

    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    print(f"입력 생성 완료: input_ids shape={input_ids.shape}")
    print(f"더미 텍스트: {dummy_text}")

    return model, input_ids, attention_mask

def main():
    """메인 디버깅 함수"""

    print("NaN 라우팅 가중치 디버깅 스크립트 시작")
    print("=" * 60)

    # 디버깅용 모델과 입력 생성
    model, input_ids, attention_mask = create_debug_model_and_inputs(
        sequence_length=32,
        batch_size=1
    )

    # NaN 디버깅 실행
    # 강제로 Expert Choice에서 토큰이 드랍되도록 설정
    # 모델 내부의 모든 라우터 설정을 변경해야 함
    print(f"DEBUG: Router 설정을 변경하여 토큰 드랍 유도 (Capacity Factor = 0.1)")
    for name, module in model.named_modules():
        if "router" in name and hasattr(module, "capacity_factor"):
             module.capacity_factor = 0.1
        if "moe" in name and hasattr(module, "expert_choice_capacity_factor"):
             module.expert_choice_capacity_factor = 0.1

    nan_layer_idx, nan_component, nan_tensor = debug_forward_with_nan_detection(
        model=model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_layers_to_check=20,  # 처음 10개 레이어만 검사
        verbose=True
    )

    # 결과 출력
    print("\n" + "=" * 60)
    print("디버깅 결과:")
    if nan_layer_idx is not None:
        traceback.print_exc()

        print(f"❌ NaN 발견!")
        print(f"레이어 인덱스: {nan_layer_idx}")
        print(f"컴포넌트: {nan_component}")
        if nan_tensor is not None:
            print(f"텐서 shape: {nan_tensor.shape}")
            print(f"NaN 개수: {torch.isnan(nan_tensor).sum().item()}")
            print(f"텐서 통계:")
            print(f"  - Mean: {nan_tensor.mean().item()}")
            print(f"  - Std: {nan_tensor.std().item()}")
            print(f"  - Min: {nan_tensor.min().item()}")
            print(f"  - Max: {nan_tensor.max().item()}")
    else:
        print(f"✅ NaN 미발견! 모든 레이어 정상 통과")

    print("=" * 60)

if __name__ == "__main__":
    main()
