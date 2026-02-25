#!/usr/bin/env python3
"""
Configuration Selector for G3MoE SFT Training
Based on Google's Gemini fine-tuning guidelines
"""

import os
import sys
import json
import argparse
from typing import Dict, Any

def analyze_dataset_characteristics(dataset_path: str = None, dataset_name: str = None) -> Dict[str, Any]:
    """
    Analyze dataset characteristics to recommend appropriate config
    """
    try:
        if dataset_name:
            # For HF Hub datasets, provide general estimates
            if dataset_name == "Gunulhona/open_m_3":
                return {
                    "estimated_size": 10000,
                    "avg_context_length": 800,
                    "recommendation": "large"
                }
            else:
                print(f"Unknown dataset: {dataset_name}")
                return {"estimated_size": 1000, "avg_context_length": 500, "recommendation": "small"}
        
        if dataset_path and os.path.exists(dataset_path):
            # Analyze local JSONL file
            import json
            examples = []
            with open(dataset_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        examples.append(json.loads(line.strip()))
                        if len(examples) >= 1000:  # Sample first 1000 for analysis
                            break
                    except:
                        continue
            
            total_examples = len(examples)
            if examples:
                # Estimate average context length
                avg_length = sum(len(str(ex)) for ex in examples) / len(examples)
                return {
                    "estimated_size": total_examples,
                    "avg_context_length": avg_length // 4,  # Rough token estimate
                    "recommendation": "large" if total_examples >= 1000 or avg_length >= 2000 else "small"
                }
    
    except Exception as e:
        print(f"Error analyzing dataset: {e}")
    
    # Default fallback
    return {"estimated_size": 1000, "avg_context_length": 500, "recommendation": "small"}

def get_config_recommendation(dataset_size: int, avg_context_length: int) -> str:
    """
    Get configuration recommendation based on Google guidelines and context length
    """
    if avg_context_length >= 50000:  # Very long context (50K+ tokens)
        return "120k_deepspeed"
    elif dataset_size < 1000 and avg_context_length < 500:
        return "small"
    else:
        return "large"

def display_recommendations():
    """
    Display Google's official recommendations
    """
    print("🎯 Google Gemini Fine-tuning Guidelines")
    print("=" * 50)
    
    print("\n📊 소규모 데이터셋 (< 1000개 예제, 평균 컨텍스트 < 500토큰):")
    print("   • Epochs: 20")
    print("   • Learning Rate: 2e-4 (10x multiplier)")
    print("   • LoRA Rank: 4")
    print("   • Max Sequence Length: 1024")
    print("   • Config: sft/config/g3moe_small_dataset_config.json")
    
    print("\n📊 대규모 데이터셋 (>= 1000개 예제 또는 평균 컨텍스트 >= 500토큰):")
    print("   • Epochs: 10")
    print("   • Learning Rate: 1e-4 (5x multiplier)")
    print("   • LoRA Rank: 8")
    print("   • Max Sequence Length: 120000")
    print("   • Config: sft/config/g3moe_large_dataset_config.json")
    
    print("\n🚀 초장문 컨텍스트 (>= 50K 토큰, DeepSpeed 필수):")
    print("   • Epochs: 5")
    print("   • Learning Rate: 5e-5")
    print("   • LoRA Rank: 16")
    print("   • Max Sequence Length: 120000")
    print("   • DeepSpeed: ZeRO-3 + CPU Offload")
    print("   • Config: sft/config/g3moe_120k_deepspeed_config.json")
    
    print("\n🔑 핵심 원칙:")
    print("   • 품질 > 양: 고품질 소규모 데이터셋이 더 효과적")
    print("   • LoRA 활용: 메모리 효율적 fine-tuning")
    print("   • 복잡한 예제 중심: 기본 모델이 어려워하는 케이스에 집중")
    print("   • 검증 데이터셋: 과적합 방지를 위한 필수 요소")

def main():
    parser = argparse.ArgumentParser(description="Select optimal G3MoE SFT configuration")
    parser.add_argument("--dataset-name", type=str, help="HuggingFace dataset name")
    parser.add_argument("--dataset-path", type=str, help="Local dataset file path")
    parser.add_argument("--dataset-size", type=int, help="Manual dataset size")
    parser.add_argument("--avg-context-length", type=int, help="Manual average context length")
    parser.add_argument("--show-guidelines", action="store_true", help="Show Google guidelines")
    
    args = parser.parse_args()
    
    if args.show_guidelines:
        display_recommendations()
        return
    
    print("🚀 G3MoE SFT Configuration Selector")
    print("=" * 40)
    
    # Analyze dataset characteristics
    if args.dataset_size and args.avg_context_length:
        analysis = {
            "estimated_size": args.dataset_size,
            "avg_context_length": args.avg_context_length,
            "recommendation": get_config_recommendation(args.dataset_size, args.avg_context_length)
        }
    else:
        analysis = analyze_dataset_characteristics(args.dataset_path, args.dataset_name)
    
    print(f"\n📊 데이터셋 분석 결과:")
    print(f"   • 예상 데이터셋 크기: {analysis['estimated_size']:,}개")
    print(f"   • 평균 컨텍스트 길이: {analysis['avg_context_length']:,}토큰")
    
    # Get recommendation
    recommendation = analysis['recommendation']
    
    print(f"\n🎯 권장 설정: {recommendation.upper()} dataset configuration")
    
    if recommendation == "small":
        config_file = "sft/config/g3moe_small_dataset_config.json"
        print(f"   • 설정 파일: {config_file}")
        print(f"   • 실행 명령: ./sft/run_g3moe_sft.sh {config_file}")
        print("\n📋 설정 세부사항:")
        print("   • Epochs: 20")
        print("   • Learning Rate: 2e-4")
        print("   • LoRA Rank: 4")
        print("   • Batch Size: 2")
        print("   • Max Sequence Length: 120000")
    elif recommendation == "120k_deepspeed":
        config_file = "sft/config/g3moe_120k_deepspeed_config.json"
        print(f"   • 설정 파일: {config_file}")
        print(f"   • 실행 명령: ./sft/run_g3moe_deepspeed.sh {config_file}")
        print("\n📋 설정 세부사항:")
        print("   • Epochs: 5")
        print("   • Learning Rate: 5e-5")
        print("   • LoRA Rank: 16")
        print("   • Batch Size: 1")
        print("   • Max Sequence Length: 120000")
        print("   • DeepSpeed: ZeRO-3 (CPU Offload)")
        print("   • GPU 요구사항: 80GB+ VRAM")
        print("   • RAM 요구사항: 64GB+ System RAM")
    else:
        config_file = "sft/config/g3moe_large_dataset_config.json"
        print(f"   • 설정 파일: {config_file}")
        print(f"   • 실행 명령: ./sft/run_g3moe_deepspeed.sh {config_file}")
        print("\n📋 설정 세부사항:")
        print("   • Epochs: 10")
        print("   • Learning Rate: 1e-4")
        print("   • LoRA Rank: 8")
        print("   • Batch Size: 1")
        print("   • Max Sequence Length: 120000")
        print("   • DeepSpeed: ZeRO-3")
    
    print(f"\n💡 설정 파일 확인:")
    if os.path.exists(config_file):
        print(f"   ✅ {config_file} 파일이 존재합니다.")
    else:
        print(f"   ❌ {config_file} 파일이 없습니다.")
        print("   먼저 설정 파일을 생성해주세요.")
    
    print(f"\n🔗 참고자료:")
    print("   • Google Gemini Fine-tuning Guide:")
    print("     https://medium.com/google-cloud/fine-tuning-gemini-best-practices-for-data-hyperparameters-and-evaluation-65f7c7b6b15f")

if __name__ == "__main__":
    main() 