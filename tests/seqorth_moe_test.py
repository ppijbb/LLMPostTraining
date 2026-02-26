"""
Example usage of Seqorth (Gram Matrix-based Specialization Routing)

This module demonstrates how to convert any pretrained HuggingFace model
into a Mixture of Experts (MoE) model using SeqorthRouter routing method.
Seqorth uses Gram matrix-based orthogonal constraints for expert specialization
and diversity.
"""
import os
import time
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from transformers import AutoModelForCausalLM, Gemma3ForConditionalGeneration, Qwen3VLForConditionalGeneration, AutoTokenizer
from models.seqorth import upcycle_model_to_moe, SeqorthRouter
from models.g3moe_model import G3MoEMLP

# Note: upcycle_model_to_moe automatically enables global routing state passing
# via the wrapper. If you need manual control, you can also use:
# from models.seqorth_wrapper import enable_global_routing_state

os.environ["CUDA_VISIBLE_DEVICES"] = "3"


def format_parameters(number):
    """Format parameter count in human-readable format"""
    if number >= 1_000_000_000:
        return f"{number / 1_000_000_000:.4f} B"
    elif number >= 1_000_000:
        return f"{number / 1_000_000:.4f} M"
    else:
        return str(number)


def count_parameters(model):
    """Count total number of trainable parameters in the model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_size_comparison(params_before, model_after, model_name):
    """Print comparison of model sizes before and after upcycling"""
    params_after = count_parameters(model_after)
    
    print(f"\n{'='*60}")
    print(f"Model Size Comparison: {model_name}")
    print(f"{'='*60}")
    print(f"Before upcycling: {format_parameters(params_before)} ({params_before:,} parameters)")
    print(f"After upcycling:  {format_parameters(params_after)} ({params_after:,} parameters)")
    
    if params_before > 0:
        ratio = params_after / params_before
        increase = params_after - params_before
        print(f"Increase:         {format_parameters(increase)} ({increase:,} parameters)")
        print(f"Size ratio:       {ratio:.4f}x ({ratio*100:.2f}% of original)")
    print(f"{'='*60}\n")


def count_active_parameters(model, sample_inputs=None, top_k=None, verbose=True):
    """
    Inference 시 실제 활성화되는 파라미터 수를 계산합니다.
    
    실제 forward pass를 실행하여 활성화된 expert를 추적합니다.
    
    Args:
        model: G3MoE 모델
        sample_inputs: 실제 forward pass를 위한 샘플 입력 (dict, optional)
                      제공되지 않으면 이론적 계산 사용
        top_k: 활성화되는 expert 수 (None이면 config에서 가져옴)
        verbose: 상세 출력 여부
    
    Returns:
        dict: 활성화 파라미터 정보
    """
    # Config에서 MoE 설정 가져오기
    if hasattr(model, 'config'):
        config = model.config
        if hasattr(config, 'text_config'):
            config = config.text_config
    else:
        raise ValueError("Model must have config attribute")
    
    n_routed_experts = getattr(config, 'n_routed_experts', 0)
    n_shared_experts = getattr(config, 'n_shared_experts', 1)
    num_experts_per_tok = top_k if top_k is not None else getattr(config, 'num_experts_per_tok', 2)
    first_k_dense_replace = getattr(config, 'first_k_dense_replace', 0)
    num_hidden_layers = getattr(config, 'num_hidden_layers', 0)
    
    # MoE 레이어 수 계산 (first_k_dense_replace 이후 레이어만 MoE)
    num_moe_layers = max(0, num_hidden_layers - first_k_dense_replace)
    
    # 전체 파라미터 카운트
    total_params = sum(p.numel() for p in model.parameters())
    
    # 카테고리별 파라미터 카운트
    embedding_params = 0
    attention_params = 0
    shared_expert_params = 0
    routed_expert_params = 0
    dense_mlp_params = 0  # Dense 레이어의 MLP (MoE가 아닌 레이어)
    router_params = 0
    global_router_params = 0  # Global router (공유됨, 한 번만 카운트)
    norm_params = 0
    lm_head_params = 0
    vision_params = 0
    other_params = 0
    
    # Global router는 한 번만 카운트해야 함
    global_router_seen = set()
    
    # 레이어 인덱스 추출 헬퍼 함수
    def get_layer_idx(name):
        """레이어 인덱스 추출 (예: 'model.layers.5.moe' -> 5)"""
        import re
        match = re.search(r'\.layers\.(\d+)\.', name)
        if match:
            return int(match.group(1))
        return -1
    
    for name, param in model.named_parameters():
        param_count = param.numel()
        layer_idx = get_layer_idx(name)
        is_moe_layer = layer_idx >= first_k_dense_replace if layer_idx >= 0 else False
        
        # Vision tower 파라미터
        if 'vision_tower' in name or 'vision_model' in name:
            vision_params += param_count
        # Embedding 파라미터
        elif 'embed' in name.lower():
            embedding_params += param_count
        # Attention 파라미터
        elif 'self_attn' in name or 'attn' in name:
            attention_params += param_count
        # Global Router 파라미터 (공유되므로 한 번만 카운트)
        elif 'global_router' in name or ('router' in name and 'global' in name and 'language_model' in name):
            # Global router는 한 번만 카운트
            router_key = 'global_router'
            if router_key not in global_router_seen:
                global_router_params += param_count
                global_router_seen.add(router_key)
        # Shared Expert 파라미터 (항상 활성화)
        elif 'shared_experts' in name or 'shared_expert' in name:
            shared_expert_params += param_count
        # Routed Expert 파라미터 (MoE 레이어의 experts만)
        elif 'experts' in name and 'shared' not in name:
            # MoE 레이어의 experts만 카운트
            if is_moe_layer or 'moe' in name.lower():
                routed_expert_params += param_count
            else:
                # Dense 레이어의 MLP는 다른 곳에서 처리
                other_params += param_count
        # Dense MLP 파라미터 (MoE가 아닌 레이어의 MLP, 항상 활성화)
        elif ('mlp' in name.lower() or 'gate_proj' in name or 'up_proj' in name or 'down_proj' in name) and \
             'moe' not in name.lower() and 'expert' not in name.lower():
            # MoE가 아닌 레이어의 MLP만 카운트
            if not is_moe_layer:
                dense_mlp_params += param_count
            else:
                other_params += param_count
        # Router 파라미터 (로컬 router, 각 레이어마다 있지만 항상 활성화)
        elif 'router' in name or 'gate' in name:
            # Global router가 아닌 경우만 카운트
            if 'global' not in name:
                router_params += param_count
        # LayerNorm 파라미터
        elif 'norm' in name.lower() or 'layernorm' in name.lower():
            norm_params += param_count
        # LM Head 파라미터
        elif 'lm_head' in name or 'score' in name:
            lm_head_params += param_count
        else:
            other_params += param_count
    
    # Forward pass에서 모든 레이어가 활성화됩니다!
    # 단, MoE 레이어 내의 Routed Experts만 sparse activation (토큰당 top_k개만)
    
    # 항상 활성화되는 파라미터 (모든 레이어)
    always_active_params = (
        embedding_params +
        attention_params +  # 모든 레이어의 attention
        shared_expert_params +  # 모든 MoE 레이어의 shared expert
        dense_mlp_params +  # Dense 레이어의 MLP (모두 활성화)
        global_router_params +  # Global router (항상 활성화)
        router_params +  # 모든 router (항상 활성화)
        norm_params +  # 모든 LayerNorm
        lm_head_params +
        vision_params +
        other_params
    )
    
    # Routed Experts만 sparse activation
    # 실제 forward pass를 통해 활성화된 expert 추적
    if sample_inputs is not None and n_routed_experts > 0:
        # 실제 forward pass로 활성화된 expert 추적
        model.eval()
        activated_experts_per_layer = {}
        
        def hook_fn(module, input, output):
            """Forward hook으로 실제 활성화된 expert 추적"""
            if hasattr(module, 'last_selected_experts'):
                selected_experts = module.last_selected_experts
                if selected_experts is not None:
                    # 실제 활성화된 expert ID 추출
                    unique_experts = torch.unique(selected_experts)
                    layer_name = f"layer_{getattr(module, 'iter', 'unknown')}"
                    activated_experts_per_layer[layer_name] = unique_experts.cpu().tolist()
        
        # Hook 등록
        hooks = []
        for name, module in model.named_modules():
            if hasattr(module, 'experts') and hasattr(module, 'num_experts'):
                # MoE 레이어에 hook 등록
                hook = module.register_forward_hook(hook_fn)
                hooks.append(hook)
        
        # Forward pass 실행
        with torch.no_grad():
            try:
                _ = model(**sample_inputs)
            except Exception as e:
                if verbose:
                    print(f"Warning: Forward pass failed: {e}. Using theoretical calculation.")
                sample_inputs = None  # Fallback to theoretical
        
        # Hook 제거
        for hook in hooks:
            hook.remove()
        
        # 실제 활성화된 expert 수 계산
        if activated_experts_per_layer:
            # 각 레이어에서 활성화된 expert 수의 평균
            total_activated = sum(len(experts) for experts in activated_experts_per_layer.values())
            avg_activated_per_layer = total_activated / len(activated_experts_per_layer) if activated_experts_per_layer else num_experts_per_tok
            activation_ratio = avg_activated_per_layer / n_routed_experts
            active_routed_expert_params = int(routed_expert_params * activation_ratio)
            actual_measurement = True
        else:
            # Fallback to theoretical
            activation_ratio = num_experts_per_tok / n_routed_experts
            active_routed_expert_params = int(routed_expert_params * activation_ratio)
            actual_measurement = False
    else:
        # 이론적 계산 (forward pass 없이)
        if n_routed_experts > 0:
            activation_ratio = num_experts_per_tok / n_routed_experts
            active_routed_expert_params = int(routed_expert_params * activation_ratio)
        else:
            activation_ratio = 0.0
            active_routed_expert_params = 0
        actual_measurement = False
    
    # 전체 활성화 파라미터 = 항상 활성화 + Routed Experts 중 활성화 부분
    active_params = always_active_params + active_routed_expert_params
    activation_rate = active_params / total_params if total_params > 0 else 0.0
    
    result = {
        'total_params': total_params,
        'active_params': active_params,
        'activation_rate': activation_rate,
        'breakdown': {
            'embedding': embedding_params,
            'attention': attention_params,
            'shared_experts': shared_expert_params,
            'dense_mlp': dense_mlp_params,
            'routed_experts_total': routed_expert_params,
            'routed_experts_active': active_routed_expert_params,
            'global_router': global_router_params,
            'router': router_params,
            'norm': norm_params,
            'lm_head': lm_head_params,
            'vision': vision_params,
            'other': other_params,
        },
        'config': {
            'n_routed_experts': n_routed_experts,
            'n_shared_experts': n_shared_experts,
            'num_experts_per_tok': num_experts_per_tok,
            'num_moe_layers': num_moe_layers,
            'first_k_dense_replace': first_k_dense_replace,
            'num_hidden_layers': num_hidden_layers,
        },
        'actual_measurement': actual_measurement if 'actual_measurement' in locals() else False
    }
    
    if verbose:
        print("\n" + "="*80)
        print("MoE Model Active Parameter Analysis")
        print("="*80)
        print(f"\n📊 Configuration:")
        print(f"  - Total Hidden Layers: {num_hidden_layers}")
        print(f"  - MoE Layers: {num_moe_layers} (starting from layer {first_k_dense_replace})")
        print(f"  - Routed Experts: {n_routed_experts}")
        print(f"  - Shared Experts per Layer: {n_shared_experts}")
        print(f"  - Active Experts per Token (top_k): {num_experts_per_tok}")
        if actual_measurement:
            print(f"  - Expert Activation Ratio: {activation_ratio:.4f} (실제 측정: 평균 {activation_ratio*n_routed_experts:.1f}개 expert 활성화)")
        else:
            print(f"  - Expert Activation Ratio: {activation_ratio:.4f} ({num_experts_per_tok}/{n_routed_experts}) [이론적 계산]")
        
        print(f"\n📈 Parameter Breakdown:")
        print(f"  Total Parameters:           {format_parameters(total_params):>15} (100.00%)")
        print(f"  Active Parameters:         {format_parameters(active_params):>15} ({activation_rate*100:.2f}%)")
        print(f"\n  Always Active Components:")
        print(f"    - Embedding:              {format_parameters(embedding_params):>15} ({embedding_params/total_params*100:.2f}%)")
        print(f"    - Attention:              {format_parameters(attention_params):>15} ({attention_params/total_params*100:.2f}%)")
        print(f"    - Dense MLP:              {format_parameters(dense_mlp_params):>15} ({dense_mlp_params/total_params*100:.2f}%)")
        print(f"    - Shared Experts:         {format_parameters(shared_expert_params):>15} ({shared_expert_params/total_params*100:.2f}%)")
        print(f"    - Global Router:          {format_parameters(global_router_params):>15} ({global_router_params/total_params*100:.2f}%)")
        print(f"    - Local Router:           {format_parameters(router_params):>15} ({router_params/total_params*100:.2f}%)")
        print(f"    - LayerNorm:               {format_parameters(norm_params):>15} ({norm_params/total_params*100:.2f}%)")
        print(f"    - LM Head:                 {format_parameters(lm_head_params):>15} ({lm_head_params/total_params*100:.2f}%)")
        print(f"    - Vision Tower:            {format_parameters(vision_params):>15} ({vision_params/total_params*100:.2f}%)")
        print(f"    - Other:                   {format_parameters(other_params):>15} ({other_params/total_params*100:.2f}%)")
        print(f"\n  Routed Experts (Sparse Activation):")
        print(f"    - Total Routed Experts:    {format_parameters(routed_expert_params):>15} ({routed_expert_params/total_params*100:.2f}%)")
        print(f"    - Active Routed Experts:  {format_parameters(active_routed_expert_params):>15} ({active_routed_expert_params/total_params*100:.2f}%)")
        if actual_measurement:
            print(f"      (실제 forward pass에서 {activation_ratio*100:.2f}% 활성화됨)")
        else:
            print(f"      (이론적: {activation_ratio*100:.2f}% 활성화, 실제 측정하려면 sample_inputs 제공 필요)")
        
        print(f"\n💡 Key Insight:")
        print(f"  During inference:")
        print(f"    - All layers are activated (attention, MLP, shared experts, router, etc.)")
        print(f"    - Only Routed Experts use sparse activation: {activation_ratio*100:.2f}% per token")
        print(f"    - Overall active parameters: {activation_rate*100:.2f}% of total")
        print(f"    - Inactive parameters: {format_parameters(total_params - active_params)}")
        print(f"    - Efficiency: {format_parameters(active_params)} active / {format_parameters(total_params)} total")
        print("="*80 + "\n")
    
    return result


def create_text_input(tokenizer, test_text="What is the capital of France?", max_length=128):
    """
    Create input from actual text using tokenizer with chat template
    
    Args:
        tokenizer: The tokenizer to use
        test_text: Test text to encode (default: simple question)
        max_length: Maximum sequence length (default: 128)
    
    Returns:
        Dictionary with input_ids and attention_mask
    """
    # Create a simple conversation format
    messages = [
        {"role": "user", "content": test_text}
    ]
    
    # Try to use apply_chat_template if available, otherwise just tokenize
    try:
        if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template is not None:
            # Use chat template
            tokenized = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
                max_length=max_length,
                truncation=True,
                return_dict=True,
                return_tensors="pt"
            )
        else:
            # Fallback to simple tokenization
            tokenized = tokenizer(
                test_text,
                max_length=max_length,
                truncation=True,
                padding=False,
                return_tensors="pt"
            )
    except Exception as e:
        # If chat template fails, fallback to simple tokenization
        print(f"Warning: Chat template failed ({e}), using simple tokenization")
        tokenized = tokenizer(
            test_text,
            max_length=max_length,
            truncation=True,
            padding=False,
            return_tensors="pt"
        )
    
    return {
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"]
    }


def measure_forward_speed(model, dummy_input, num_runs=10, warmup_runs=3, device=None):
    """
    Measure forward pass speed for a model
    
    Args:
        model: The model to test
        dummy_input: Dictionary with input_ids and attention_mask
        num_runs: Number of runs to average (default: 10)
        warmup_runs: Number of warmup runs (default: 3)
        device: Device to run on (None = auto-detect from model)
    
    Returns:
        Dictionary with timing statistics
    """
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    model.to(device)
    
    # Move inputs to device
    input_ids = dummy_input["input_ids"].to(device)
    attention_mask = dummy_input["attention_mask"].to(device)
    
    # Warmup runs
    with torch.inference_mode():
        for _ in range(warmup_runs):
            _ = model(input_ids=input_ids, attention_mask=attention_mask)
    
    # Synchronize CUDA if using GPU
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Actual measurement
    times = []
    with torch.inference_mode():
        for _ in range(num_runs):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            start_time = time.time()
            _ = model(input_ids=input_ids, attention_mask=attention_mask)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            end_time = time.time()
            times.append(end_time - start_time)
    
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    std_time = (sum((t - avg_time) ** 2 for t in times) / len(times)) ** 0.5
    
    # Calculate throughput (tokens per second)
    seq_len = input_ids.shape[1]
    batch_size = input_ids.shape[0]
    throughput = (seq_len * batch_size) / avg_time
    
    return {
        "avg_time": avg_time,
        "min_time": min_time,
        "max_time": max_time,
        "std_time": std_time,
        "throughput": throughput,
        "times": times
    }


def measure_generation_speed(model, tokenizer, dummy_input, max_new_tokens=32, num_runs=10, warmup_runs=3, device=None):
    """
    Measure generation speed (forward + generate) for a model
    
    Args:
        model: The model to test
        tokenizer: The tokenizer
        dummy_input: Dictionary with input_ids and attention_mask
        max_new_tokens: Maximum number of new tokens to generate (default: 32)
        num_runs: Number of runs to average (default: 10)
        warmup_runs: Number of warmup runs (default: 3)
        device: Device to run on (None = auto-detect from model)
    
    Returns:
        Dictionary with timing statistics
    """
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    model.to(device)
    
    # Move inputs to device
    input_ids = dummy_input["input_ids"].to(device)
    attention_mask = dummy_input["attention_mask"].to(device)
    
    # Warmup runs
    with torch.inference_mode():
        for _ in range(warmup_runs):
            _ = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
            )
    
    # Synchronize CUDA if using GPU
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Actual measurement
    times = []
    generated_tokens = []
    generated_text = None  # Store single generated text sample (deterministic, so all runs are same)
    
    with torch.inference_mode():
        for run_idx in range(num_runs):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            start_time = time.time()
            output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
            )
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            end_time = time.time()
            times.append(end_time - start_time)
            
            # Count generated tokens (excluding input)
            num_generated = output.shape[1] - input_ids.shape[1]
            generated_tokens.append(num_generated)
            
            # Decode generated text (only store once, deterministic so all runs produce same output)
            if run_idx == 0:
                generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    std_time = (sum((t - avg_time) ** 2 for t in times) / len(times)) ** 0.5
    
    # Calculate throughput (tokens per second)
    avg_generated_tokens = sum(generated_tokens) / len(generated_tokens)
    throughput = avg_generated_tokens / avg_time if avg_time > 0 else 0
    
    return {
        "avg_time": avg_time,
        "min_time": min_time,
        "max_time": max_time,
        "std_time": std_time,
        "throughput": throughput,
        "avg_generated_tokens": avg_generated_tokens,
        "times": times,
        "generated_tokens": generated_tokens,
        "generated_text": generated_text  # Store single generated text (deterministic)
    }


def print_speed_comparison(speed_before, speed_after, model_name, is_generation=False):
    """Print comparison of speeds before and after upcycling (horizontal layout)"""
    # Calculate speed change metrics
    if speed_before['avg_time'] > 0:
        speed_ratio = speed_before['avg_time'] / speed_after['avg_time']
        time_diff = (speed_after['avg_time'] - speed_before['avg_time']) * 1000
        time_diff_percent = (time_diff / (speed_before['avg_time'] * 1000)) * 100
    
    # Format values
    before_avg = f"{speed_before['avg_time']*1000:.2f} ms"
    before_minmax = f"{speed_before['min_time']*1000:.2f} ms / {speed_before['max_time']*1000:.2f} ms"
    before_std = f"{speed_before['std_time']*1000:.2f} ms"
    before_throughput = f"{speed_before['throughput']:.2f} tokens/sec"
    
    after_avg = f"{speed_after['avg_time']*1000:.2f} ms"
    after_minmax = f"{speed_after['min_time']*1000:.2f} ms / {speed_after['max_time']*1000:.2f} ms"
    after_std = f"{speed_after['std_time']*1000:.2f} ms"
    after_throughput = f"{speed_after['throughput']:.2f} tokens/sec"
    
    if speed_before['avg_time'] > 0:
        change_ratio = f"{speed_ratio:.4f}x"
        if time_diff > 0:
            change_text = f"Slower by: {time_diff:.2f} ms ({time_diff_percent:.2f}%)"
        else:
            change_text = f"Faster by: {abs(time_diff):.2f} ms ({abs(time_diff_percent):.2f}%)"
    else:
        change_ratio = "N/A"
        change_text = "N/A"
    
    # Add generation-specific metrics if available
    if is_generation:
        before_tokens = f"{speed_before.get('avg_generated_tokens', 0):.1f} tokens"
        after_tokens = f"{speed_after.get('avg_generated_tokens', 0):.1f} tokens"
    
    # Print horizontal comparison table
    title = "Generation Speed Comparison" if is_generation else "Forward Speed Comparison"
    
    # Define column widths for consistent alignment
    col1_width = 20
    col2_width = 30
    col3_width = 30
    separator = " │ "
    
    print(f"\n{'='*90}")
    print(f"{title}: {model_name}".center(90))
    print(f"{'='*90}")
    
    # Header row
    header = f"{'Metric':<{col1_width}}{separator}{'Before upcycling':<{col2_width}}{separator}{'After upcycling':<{col3_width}}"
    print(header)
    
    # Separator row - exactly match header width
    sep_row = f"{'─'*col1_width}──{'─'*col2_width}──{'─'*col3_width}"
    print(sep_row)
    
    # Data rows
    print(f"{'Avg time':<{col1_width}}{separator}{before_avg:<{col2_width}}{separator}{after_avg:<{col3_width}}")
    print(f"{'Min/Max':<{col1_width}}{separator}{before_minmax:<{col2_width}}{separator}{after_minmax:<{col3_width}}")
    print(f"{'Std dev':<{col1_width}}{separator}{before_std:<{col2_width}}{separator}{after_std:<{col3_width}}")
    if is_generation:
        print(f"{'Avg tokens':<{col1_width}}{separator}{before_tokens:<{col2_width}}{separator}{after_tokens:<{col3_width}}")
    print(f"{'Throughput':<{col1_width}}{separator}{before_throughput:<{col2_width}}{separator}{after_throughput:<{col3_width}}")
    
    # Bottom separator
    print(sep_row)
    
    # Summary rows
    print(f"{'Speed ratio':<{col1_width}}{separator}{change_ratio:<{col2_width}}{separator}{'':<{col3_width}}")
    print(f"{'Change':<{col1_width}}{separator}{change_text:<{col2_width}}{separator}{'':<{col3_width}}")
    print(f"{'='*90}")
    
    # Print generated text samples if available
    if is_generation:
        before_text = speed_before.get('generated_text', None)
        after_text = speed_after.get('generated_text', None)
        
        if before_text or after_text:
            print(f"\n{'Generated Text Comparison':<90}")
            print(f"{'─'*90}")
            
            if before_text:
                print(f"\nBefore upcycling:")
                # Wrap long text for better readability
                text_lines = []
                max_line_len = 88
                words = before_text.split()
                current_line = ""
                for word in words:
                    if len(current_line) + len(word) + 1 <= max_line_len:
                        current_line += (word + " ") if current_line else word
                    else:
                        if current_line:
                            text_lines.append("  " + current_line)
                        current_line = word
                if current_line:
                    text_lines.append("  " + current_line)
                
                for line in text_lines:
                    print(line)
            
            if after_text:
                print(f"\nAfter upcycling:")
                # Wrap long text for better readability
                text_lines = []
                max_line_len = 88
                words = after_text.split()
                current_line = ""
                for word in words:
                    if len(current_line) + len(word) + 1 <= max_line_len:
                        current_line += (word + " ") if current_line else word
                    else:
                        if current_line:
                            text_lines.append("  " + current_line)
                        current_line = word
                if current_line:
                    text_lines.append("  " + current_line)
                
                for line in text_lines:
                    print(line)
            
            print(f"{'─'*90}\n")


def example_llama_to_moe():
    """Example: Convert Llama 3.2 to MoE"""
    print("Loading Llama 3.2 model...")
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    model_before = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Count parameters before upcycling
    params_before = count_parameters(model_before)
    print(f"Model loaded. Parameters before upcycling: {format_parameters(params_before)}")
    
    # Measure forward speed before upcycling
    print("Measuring forward speed before upcycling...")
    text_input = create_text_input(tokenizer, test_text="What is the capital of France?", max_length=128)
    speed_before = measure_forward_speed(model_before, text_input, num_runs=10, warmup_runs=3)
    
    # Measure generation speed BEFORE upcycling (must be done before upcycling modifies model in-place)
    print("Measuring generation speed before upcycling (max_new_tokens=32)...")
    gen_speed_before = measure_generation_speed(model_before, tokenizer, text_input, max_new_tokens=32, num_runs=5, warmup_runs=2)
    
    # Define MoE configuration
    moe_config = {
        "hidden_size": model_before.config.hidden_size,
        "intermediate_size": model_before.config.intermediate_size,
        "num_experts": 2,
        "num_experts_per_tok": 1,  # Must be <= num_experts
        "router_dim": 128,
        "n_shared_experts": 1,
        "first_k_dense_replace": 2,  # Keep first 2 layers as dense
        "router_jitter_noise": 0.01,
        "input_jitter_noise": 0.0,
        "freeze_shared_experts": True,
        "balancing_strength": 0.01,
        "ema_alpha": 0.99,
        "hidden_activation": "silu",
    }
    
    print("Upcycling Llama 3.2 to MoE...")
    model_after = upcycle_model_to_moe(
        model_before, 
        moe_config,
        expert_module_class=G3MoEMLP,
        layer_start_idx=2,  # Start from layer 2
        verbose=True
    )
    
    print("Conversion complete!")
    print_size_comparison(params_before, model_after, "Llama 3.2")
    
    # Prepare input for testing
    text_input_after = create_text_input(tokenizer, test_text="What is the capital of France?", max_length=128)
    
    # Measure active parameters during inference
    print("\nMeasuring active parameters during inference...")
    active_param_info = count_active_parameters(model_after, sample_inputs=text_input_after, verbose=True)
    
    # Measure forward speed after upcycling
    print("Measuring forward speed after upcycling...")
    speed_after = measure_forward_speed(model_after, text_input_after, num_runs=10, warmup_runs=3)
    print_speed_comparison(speed_before, speed_after, "Llama 3.2")
    
    # Measure generation speed AFTER upcycling
    print("Measuring generation speed after upcycling (max_new_tokens=32)...")
    gen_speed_after = measure_generation_speed(model_after, tokenizer, text_input_after, max_new_tokens=32, num_runs=5, warmup_runs=2)
    print_speed_comparison(gen_speed_before, gen_speed_after, "Llama 3.2", is_generation=True)
    
    return model_after


def example_gemma3_to_moe():
    """Example: Convert Gemma 3 to MoE"""
    print("Loading Gemma 3 model...")
    model_name = "google/gemma-3-4b-it"
    model_before = Gemma3ForConditionalGeneration.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Count parameters before upcycling
    params_before = count_parameters(model_before)
    print(f"Model loaded. Parameters before upcycling: {format_parameters(params_before)}")
    
    # Measure forward speed before upcycling
    print("Measuring forward speed before upcycling...")
    text_input = create_text_input(tokenizer, test_text="What is the capital of France?", max_length=128)
    speed_before = measure_forward_speed(model_before, text_input, num_runs=10, warmup_runs=3)
    
    # Measure generation speed BEFORE upcycling (must be done before upcycling modifies model in-place)
    print("Measuring generation speed before upcycling (max_new_tokens=32)...")
    gen_speed_before = measure_generation_speed(model_before, tokenizer, text_input, max_new_tokens=32, num_runs=5, warmup_runs=2)
    
    # Define MoE configuration
    moe_config = {
        "hidden_size": model_before.config.text_config.hidden_size,
        "intermediate_size": model_before.config.text_config.intermediate_size,
        "num_experts": 3,
        "num_experts_per_tok": 1,  # Must be <= num_experts
        "router_dim": 128,
        "n_shared_experts": 1,
        "first_k_dense_replace": 2,  # Keep first 2 layers as dense
        "router_jitter_noise": 0.01,
        "input_jitter_noise": 0.0,
        "freeze_shared_experts": True,
        "balancing_strength": 0.01,
        "ema_alpha": 0.99,
        "hidden_activation": "gelu",
    }
    
    print("Upcycling Gemma 3 to MoE...")
    model_after = upcycle_model_to_moe(
        model_before,
        moe_config,
        expert_module_class=G3MoEMLP,
        layer_start_idx=2,  # Start from layer 2
        verbose=True
    )
    
    print("Conversion complete!")
    print_size_comparison(params_before, model_after, "Gemma 3")
    
    # Prepare input for testing
    text_input_after = create_text_input(tokenizer, test_text="What is the capital of France?", max_length=128)
    
    # Measure active parameters during inference
    print("\nMeasuring active parameters during inference...")
    active_param_info = count_active_parameters(model_after, sample_inputs=text_input_after, verbose=True)
    
    # Measure forward speed after upcycling
    print("Measuring forward speed after upcycling...")
    speed_after = measure_forward_speed(model_after, text_input_after, num_runs=10, warmup_runs=3)
    print_speed_comparison(speed_before, speed_after, "Gemma 3")
    
    # Measure generation speed AFTER upcycling
    print("Measuring generation speed after upcycling (max_new_tokens=32)...")
    gen_speed_after = measure_generation_speed(model_after, tokenizer, text_input_after, max_new_tokens=32, num_runs=5, warmup_runs=2)
    print_speed_comparison(gen_speed_before, gen_speed_after, "Gemma 3", is_generation=True)
    
    return model_after


def example_qwen3_to_moe():
    """Example: Convert Qwen 3 VL to MoE"""
    print("Loading Qwen 3 VL model...")
    model_name = "Qwen/Qwen3-VL-4B-Instruct"
    model_before = Qwen3VLForConditionalGeneration.from_pretrained(model_name, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Count parameters before upcycling
    params_before = count_parameters(model_before)
    print(f"Model loaded. Parameters before upcycling: {format_parameters(params_before)}")
    
    # Measure forward speed before upcycling
    print("Measuring forward speed before upcycling...")
    text_input = create_text_input(tokenizer, test_text="What is the capital of France?", max_length=128)
    speed_before = measure_forward_speed(model_before, text_input, num_runs=10, warmup_runs=3)
    
    # Measure generation speed BEFORE upcycling (must be done before upcycling modifies model in-place)
    print("Measuring generation speed before upcycling (max_new_tokens=32)...")
    gen_speed_before = measure_generation_speed(model_before, tokenizer, text_input, max_new_tokens=32, num_runs=5, warmup_runs=2)
    
    # Define MoE configuration
    moe_config = {
        "hidden_size": model_before.config.text_config.hidden_size,
        "intermediate_size": model_before.config.text_config.intermediate_size,
        "num_experts": 3,
        "num_experts_per_tok": 1,  # Must be <= num_experts
        "router_dim": 128,
        "n_shared_experts": 1,
        "first_k_dense_replace": 3,  # Keep first 3 layers as dense
        "router_jitter_noise": 0.01,
        "input_jitter_noise": 0.0,
        "freeze_shared_experts": True,
        "balancing_strength": 0.01,
        "ema_alpha": 0.99,
        "hidden_activation": "silu",
    }
    
    print("Upcycling Qwen3 VL to MoE...")
    model_after = upcycle_model_to_moe(
        model_before,
        moe_config,
        expert_module_class=G3MoEMLP,
        layer_start_idx=3,  # Start from layer 3
        verbose=True
    )
    
    print("Conversion complete!")
    print_size_comparison(params_before, model_after, "Qwen3 VL")
    
    # Measure active parameters during inference
    print("\nMeasuring active parameters during inference...")
    text_input_after = create_text_input(tokenizer, test_text="What is the capital of France?", max_length=128)
    active_param_info = count_active_parameters(model_after, sample_inputs=text_input_after, verbose=True)
    
    # Measure forward speed after upcycling
    print("Measuring forward speed after upcycling...")
    speed_after = measure_forward_speed(model_after, text_input_after, num_runs=10, warmup_runs=3)
    print_speed_comparison(speed_before, speed_after, "Qwen3 VL")
    
    # Measure generation speed AFTER upcycling
    print("Measuring generation speed after upcycling (max_new_tokens=32)...")
    gen_speed_after = measure_generation_speed(model_after, tokenizer, text_input_after, max_new_tokens=32, num_runs=5, warmup_runs=2)
    print_speed_comparison(gen_speed_before, gen_speed_after, "Qwen3 VL", is_generation=True)
    
    return model_after


def example_gpt_oss_to_moe():
    """Example: Convert GPT-OSS to MoE"""
    print("Loading GPT-OSS model...")
    model_name = "openai/gpt-oss-20b"
    # Try OpenAI GPT-OSS model, fallback to GPT-2 if not available
    try:
        model_before = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except:
        print("GPT-OSS model not found. Skipping this example.")
        return None
    
    # Count parameters before upcycling
    params_before = count_parameters(model_before)
    print(f"Model loaded. Parameters before upcycling: {format_parameters(params_before)}")
    
    # Measure forward speed before upcycling
    print("Measuring forward speed before upcycling...")
    text_input = create_text_input(tokenizer, test_text="What is the capital of France?", max_length=128)
    speed_before = measure_forward_speed(model_before, text_input, num_runs=10, warmup_runs=3)
    
    # Measure generation speed BEFORE upcycling (must be done before upcycling modifies model in-place)
    print("Measuring generation speed before upcycling (max_new_tokens=32)...")
    gen_speed_before = measure_generation_speed(model_before, tokenizer, text_input, max_new_tokens=32, num_runs=5, warmup_runs=2)
    
    # Define MoE configuration
    moe_config = {
        "hidden_size": model_before.config.n_embd if hasattr(model_before.config, 'n_embd') else model_before.config.hidden_size,
        "intermediate_size": model_before.config.n_inner if hasattr(model_before.config, 'n_inner') else model_before.config.intermediate_size,
        "num_experts": 32,
        "num_experts_per_tok": 4, # oss 20B config ref
        "router_dim": 128,
        "n_shared_experts": 1,
        "first_k_dense_replace": 0,  # Convert all layers
        "router_jitter_noise": 0.01,
        "input_jitter_noise": 0.0,
        "freeze_shared_experts": True,
        "balancing_strength": 0.01,
        "ema_alpha": 0.99,
        "hidden_activation": "gelu",
    }
    
    print("Upcycling GPT-OSS to MoE...")
    model_after = upcycle_model_to_moe(
        model_before,
        moe_config,
        expert_module_class=G3MoEMLP,
        verbose=True
    )
    
    print("Conversion complete!")
    print_size_comparison(params_before, model_after, "GPT-OSS")
    
    # Measure active parameters during inference
    print("\nMeasuring active parameters during inference...")
    text_input_after = create_text_input(tokenizer, test_text="What is the capital of France?", max_length=128)
    active_param_info = count_active_parameters(model_after, sample_inputs=text_input_after, verbose=True)
    
    # Measure forward speed after upcycling
    print("Measuring forward speed after upcycling...")
    speed_after = measure_forward_speed(model_after, text_input_after, num_runs=10, warmup_runs=3)
    print_speed_comparison(speed_before, speed_after, "GPT-OSS")
    
    # Measure generation speed AFTER upcycling
    print("Measuring generation speed after upcycling (max_new_tokens=32)...")
    gen_speed_after = measure_generation_speed(model_after, tokenizer, text_input_after, max_new_tokens=32, num_runs=5, warmup_runs=2)
    print_speed_comparison(gen_speed_before, gen_speed_after, "GPT-OSS", is_generation=True)
    
    return model_after


if __name__ == "__main__":
    print("=" * 60)
    print("Seqorth Upcycling Examples")
    print("Expression Orthogonal Routing for MoE")
    print("Using latest models: Qwen 3, Gemma 3, Llama 3.2, GPT-OSS")
    print("=" * 60)
    
    # Run examples with latest models (all <= 7B)
    print("\n1. Llama 3.2 (1B) to MoE:")
    print("-" * 60)
    model = example_llama_to_moe()
    del model
    
    print("\n2. Gemma 3 (4B) to MoE:")
    print("-" * 60)
    model = example_gemma3_to_moe()
    del model
    
    print("\n3. Qwen 3 VL (4B) to MoE:")
    print("-" * 60)
    model = example_qwen3_to_moe()
    del model
    
    # print("\n4. GPT-OSS (20B) to MoE:")
    # print("-" * 60)
    # model = example_gpt_oss_to_moe()
    # del model
    
    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)
