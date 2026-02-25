#!/usr/bin/env python3
"""
Active Parameter Comparison Tool

공정한 비교를 위해 같은 기준으로 계산:
1. Vision Tower 포함/제외 옵션
2. Dense layers vs MoE layers 구분
3. 실제 inference 측정 vs 이론적 계산
"""

import os
import sys
import torch
import argparse
import json
from pathlib import Path
from typing import Dict, Any, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from test.model_load_test import count_active_parameters


def calculate_active_params_standardized(
    model,
    exclude_vision: bool = True,
    exclude_embedding: bool = False,
    top_k: Optional[int] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    표준화된 active parameter 계산
    
    Args:
        model: 모델
        exclude_vision: Vision Tower 제외 여부
        exclude_embedding: Embedding 제외 여부
        top_k: 실제 top_k (None이면 config에서 가져옴)
        verbose: 상세 출력
        
    Returns:
        계산 결과 딕셔너리
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
    
    # 전체 파라미터 카운트
    total_params = 0
    embedding_params = 0
    attention_params = 0
    shared_expert_params = 0
    routed_expert_params = 0
    dense_mlp_params = 0
    router_params = 0
    global_router_params = 0
    norm_params = 0
    lm_head_params = 0
    vision_params = 0
    other_params = 0
    
    # 각 파라미터 분류
    for name, param in model.named_parameters():
        param_count = param.numel()
        total_params += param_count
        
        # Vision Tower
        if 'vision' in name.lower() or 'vit' in name.lower():
            if not exclude_vision:
                vision_params += param_count
            continue
        
        # Embedding
        if 'embed' in name.lower():
            if not exclude_embedding:
                embedding_params += param_count
            continue
        
        # Attention
        if 'attn' in name.lower() or 'attention' in name.lower():
            attention_params += param_count
            continue
        
        # Shared Experts
        if 'shared_expert' in name.lower() or ('shared' in name.lower() and 'expert' in name.lower()):
            shared_expert_params += param_count
            continue
        
        # Routed Experts
        if 'experts' in name.lower() and 'shared' not in name.lower():
            routed_expert_params += param_count
            continue
        
        # Dense MLP (MoE가 아닌 레이어)
        layer_idx = None
        if 'layers' in name:
            try:
                parts = name.split('.')
                for i, part in enumerate(parts):
                    if part == 'layers' and i + 1 < len(parts):
                        layer_idx = int(parts[i + 1])
                        break
            except:
                pass
        
        is_moe_layer = (layer_idx is not None and layer_idx >= first_k_dense_replace) if layer_idx is not None else False
        
        if ('mlp' in name.lower() or 'gate_proj' in name or 'up_proj' in name or 'down_proj' in name) and \
           'moe' not in name.lower() and 'expert' not in name.lower():
            if not is_moe_layer:
                dense_mlp_params += param_count
            else:
                other_params += param_count
            continue
        
        # Router
        if 'router' in name.lower() or 'gate' in name.lower():
            if 'global' in name.lower():
                global_router_params += param_count
            else:
                router_params += param_count
            continue
        
        # LayerNorm
        if 'norm' in name.lower() or 'layernorm' in name.lower():
            norm_params += param_count
            continue
        
        # LM Head
        if 'lm_head' in name.lower() or 'score' in name.lower():
            lm_head_params += param_count
            continue
        
        other_params += param_count
    
    # 항상 활성화되는 파라미터
    always_active = (
        (embedding_params if not exclude_embedding else 0) +
        attention_params +
        shared_expert_params +
        dense_mlp_params +
        global_router_params +
        router_params +
        norm_params +
        lm_head_params +
        (vision_params if not exclude_vision else 0) +
        other_params
    )
    
    # Routed Experts는 sparse activation
    expert_activation_ratio = num_experts_per_tok / n_routed_experts if n_routed_experts > 0 else 0
    active_routed_experts = routed_expert_params * expert_activation_ratio
    
    # 전체 활성화 파라미터
    active_params = always_active + active_routed_experts
    active_ratio = active_params / total_params if total_params > 0 else 0
    
    result = {
        "total_params": total_params,
        "active_params": active_params,
        "active_ratio": active_ratio,
        "breakdown": {
            "always_active": always_active,
            "routed_experts_total": routed_expert_params,
            "routed_experts_active": active_routed_experts,
            "expert_activation_ratio": expert_activation_ratio,
        },
        "components": {
            "embedding": embedding_params,
            "attention": attention_params,
            "shared_experts": shared_expert_params,
            "routed_experts": routed_expert_params,
            "dense_mlp": dense_mlp_params,
            "router": router_params,
            "global_router": global_router_params,
            "norm": norm_params,
            "lm_head": lm_head_params,
            "vision": vision_params,
            "other": other_params,
        },
        "config": {
            "n_routed_experts": n_routed_experts,
            "n_shared_experts": n_shared_experts,
            "num_experts_per_tok": num_experts_per_tok,
            "first_k_dense_replace": first_k_dense_replace,
            "num_hidden_layers": num_hidden_layers,
        }
    }
    
    if verbose:
        print(f"\n{'='*80}")
        print("Standardized Active Parameter Analysis")
        print(f"{'='*80}")
        print(f"\n📊 Configuration:")
        print(f"  - Routed Experts: {n_routed_experts}")
        print(f"  - Top-K (experts per token): {num_experts_per_tok}")
        print(f"  - Expert Activation Ratio: {expert_activation_ratio:.4f} ({num_experts_per_tok}/{n_routed_experts})")
        print(f"  - Vision Tower: {'Included' if not exclude_vision else 'Excluded'}")
        print(f"  - Embedding: {'Included' if not exclude_embedding else 'Excluded'}")
        
        print(f"\n📈 Parameter Breakdown:")
        print(f"  Total Parameters: {total_params / 1e9:.4f} B")
        print(f"  Active Parameters: {active_params / 1e9:.4f} B ({active_ratio * 100:.2f}%)")
        
        print(f"\n  Always Active Components:")
        print(f"    - Embedding: {embedding_params / 1e6:.4f} M" if not exclude_embedding else "    - Embedding: (excluded)")
        print(f"    - Attention: {attention_params / 1e6:.4f} M")
        print(f"    - Shared Experts: {shared_expert_params / 1e9:.4f} B")
        print(f"    - Dense MLP: {dense_mlp_params / 1e6:.4f} M")
        print(f"    - Router: {(router_params + global_router_params) / 1e6:.4f} M")
        print(f"    - Vision Tower: {vision_params / 1e6:.4f} M" if not exclude_vision else "    - Vision Tower: (excluded)")
        
        print(f"\n  Routed Experts (Sparse Activation):")
        print(f"    - Total: {routed_expert_params / 1e9:.4f} B")
        print(f"    - Active: {active_routed_experts / 1e9:.4f} B ({expert_activation_ratio * 100:.2f}%)")
        
        print(f"\n💡 Key Insight:")
        print(f"  Active ratio: {active_ratio * 100:.2f}%")
        print(f"  This is {'VERY EFFICIENT' if active_ratio < 0.20 else 'EFFICIENT' if active_ratio < 0.35 else 'MODERATE' if active_ratio < 0.50 else 'HIGH'}")
        
        # GPT-OSS 20B 비교 (3.4B active 가정)
        if total_params > 15e9:  # Large model
            gpt_oss_20b_active = 3.4e9
            gpt_oss_20b_total = 20e9
            gpt_oss_ratio = gpt_oss_20b_active / gpt_oss_20b_total
            
            print(f"\n📊 Comparison with GPT-OSS 20B:")
            print(f"  GPT-OSS 20B: {gpt_oss_20b_active / 1e9:.1f}B active / {gpt_oss_20b_total / 1e9:.1f}B total = {gpt_oss_ratio * 100:.2f}%")
            print(f"  This Model: {active_params / 1e9:.4f}B active / {total_params / 1e9:.4f}B total = {active_ratio * 100:.2f}%")
            
            if active_ratio < gpt_oss_ratio:
                print(f"  ✅ This model is MORE efficient than GPT-OSS 20B!")
            elif active_ratio > gpt_oss_ratio * 1.2:
                print(f"  ⚠️  This model uses more active parameters than GPT-OSS 20B")
            else:
                print(f"  ≈ Similar efficiency to GPT-OSS 20B")
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Compare Active Parameters")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to model",
    )
    parser.add_argument(
        "--exclude_vision",
        action="store_true",
        help="Exclude vision tower from calculation",
    )
    parser.add_argument(
        "--exclude_embedding",
        action="store_true",
        help="Exclude embedding from calculation",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=None,
        help="Override top_k (experts per token)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Save results to JSON file",
    )
    
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model from {args.model_path}...")
    from transformers import AutoModelForCausalLM, AutoModelForConditionalGeneration
    
    try:
        model = AutoModelForConditionalGeneration.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )
    except:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )
    
    # Calculate
    result = calculate_active_params_standardized(
        model,
        exclude_vision=args.exclude_vision,
        exclude_embedding=args.exclude_embedding,
        top_k=args.top_k,
        verbose=True,
    )
    
    # Save if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()

