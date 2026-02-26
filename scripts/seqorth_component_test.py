#!/usr/bin/env python3
"""
Seqorth 컴포넌트별 자동 점검 스크립트
각 컴포넌트를 비활성화하면서 backward 에러 발생 여부를 확인합니다.
실제 학습 대신 단일 forward/backward 테스트만 실행하여 빠르게 점검합니다.
"""

import os
import sys
import json
import time
import subprocess
import traceback
from datetime import datetime
from pathlib import Path

# 작업 디렉토리 설정
WORKSPACE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# 테스트 설정
TEST_CONFIG = {
    "timeout_seconds": 600,  # 10분 타임아웃
    "report_path": f"{WORKSPACE}/seqorth_component_report.md",
}

# 테스트할 컴포넌트 목록
COMPONENT_TESTS = [
    {
        "name": "baseline_no_seqorth",
        "description": "Seqorth 완전 비활성화 (Qwen3 원래 MoE)",
        "env_vars": {"SEQORTH_DISABLE_ALL": "1"},
    },
    {
        "name": "disable_expert_dispatch",
        "description": "Expert dispatch 비활성화 (hidden_states 그대로 반환)",
        "env_vars": {"SEQORTH_DISABLE_EXPERT_DISPATCH": "1"},
    },
    {
        "name": "disable_router",
        "description": "SeqorthRouter 비활성화 (uniform routing)",
        "env_vars": {"SEQORTH_DISABLE_ROUTER": "1"},
    },
    {
        "name": "disable_shared_experts",
        "description": "shared_experts 처리 비활성화",
        "env_vars": {"SEQORTH_DISABLE_SHARED_EXPERTS": "1"},
    },
    {
        "name": "disable_intent_gated",
        "description": "IntentGatedContextCell 비활성화",
        "env_vars": {"SEQORTH_DISABLE_INTENT_GATED": "1"},
    },
    {
        "name": "disable_expression_proj",
        "description": "ExpressionProjector 비활성화",
        "env_vars": {"SEQORTH_DISABLE_EXPRESSION_PROJ": "1"},
    },
    {
        "name": "full_seqorth",
        "description": "Seqorth 전체 활성화 (현재 상태)",
        "env_vars": {},
    },
]

# 빠른 테스트용 Python 스크립트
QUICK_TEST_SCRIPT = '''
import os
import sys
import torch
import json

# 환경 변수 설정
os.environ["SEQORTH_TEST_MODE"] = "1"
{env_setup}

# DeepSpeed/Accelerate 관련 환경 변수
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

print(f"[TEST] Starting component test: {test_name}")
print(f"[TEST] Environment: {{k: v for k, v in os.environ.items() if 'Seqorth' in k}}")

try:
    # Run with PYTHONPATH={workspace}
    from transformers import AutoConfig, AutoModelForCausalLM
    from models.seqorth_model import SeqorthExoskeletonMoEInjector, SeqorthTextConfig
    import deepspeed
    
    print("[TEST] Loading model config...")
    
    # Config 로드
    config_path = "{workspace}/config/seqorth/seqorth_qwen_config.json"
    with open(config_path) as f:
        full_config = json.load(f)
    
    model_name = full_config["model_config"]["model_name_or_path"]
    
    # 모델 로드 (ZeRO-3 사용)
    print("[TEST] Loading model with DeepSpeed ZeRO-3...")
    
    ds_config = {{
        "train_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "fp16": {{"enabled": False}},
        "bf16": {{"enabled": True}},
        "zero_optimization": {{
            "stage": 3,
            "offload_param": {{
                "device": "cpu",
                "pin_memory": True
            }},
            "overlap_comm": True,
            "contiguous_gradients": True,
            "reduce_bucket_size": 5e7,
            "stage3_prefetch_bucket_size": 5e7,
            "stage3_param_persistence_threshold": 1e5
        }}
    }}
    
    # DeepSpeed 초기화
    deepspeed.init_distributed()
    
    with deepspeed.zero.Init(config_dict_or_path=ds_config):
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            config=config,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
    
    # Seqorth injection (환경 변수에 따라 비활성화될 수 있음)
    if os.environ.get("SEQORTH_DISABLE_ALL") != "1":
        print("[TEST] Injecting Seqorth...")
        seqorth_config = SeqorthTextConfig(**full_config["model_config"]["seqorth_params"])
        seqorth_config.hidden_size = config.text_config.hidden_size
        injector = SeqorthExoskeletonMoEInjector(seqorth_config)
        model = injector.inject(model)
    else:
        print("[TEST] Seqorth disabled, using original Qwen3 MoE")
    
    # DeepSpeed 엔진 초기화
    model_engine, _, _, _ = deepspeed.initialize(
        model=model,
        config=ds_config,
        model_parameters=model.parameters()
    )
    
    model_engine.train()
    
    # 더미 입력 생성
    print("[TEST] Creating dummy input...")
    batch_size = 1
    seq_len = 128  # 짧은 시퀀스로 테스트
    
    input_ids = torch.randint(0, 1000, (batch_size, seq_len), device="cuda")
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    
    # Forward pass
    print("[TEST] Running forward pass...")
    outputs = model_engine(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
    )
    loss = outputs.loss
    print(f"[TEST] Forward pass completed. Loss: {{loss.item():.4f}}")
    
    # Backward pass
    print("[TEST] Running backward pass...")
    model_engine.backward(loss)
    print("[TEST] Backward pass completed successfully!")
    
    # 성공
    print("[TEST] ✅ TEST PASSED")
    sys.exit(0)
    
except Exception as e:
    print(f"[TEST] ❌ TEST FAILED: {{type(e).__name__}}: {{str(e)}}")
    import traceback
    traceback.print_exc()
    
    # 특정 에러 체크
    error_str = str(e)
    if "size of tensor a (0)" in error_str and "size of tensor b (2048)" in error_str:
        print("[TEST] ERROR_TYPE: tensor_size_mismatch_0_vs_2048")
    elif "CUDA out of memory" in error_str or "OutOfMemoryError" in error_str:
        print("[TEST] ERROR_TYPE: cuda_oom")
    else:
        print(f"[TEST] ERROR_TYPE: other")
    
    sys.exit(1)
'''


def run_test(test_config: dict) -> dict:
    """단일 테스트 실행"""
    result = {
        "name": test_config["name"],
        "description": test_config["description"],
        "status": "unknown",
        "error_message": None,
        "error_type": None,
        "duration_seconds": 0,
        "start_time": datetime.now().isoformat(),
    }
    
    # 환경 변수 설정 코드 생성
    env_setup_lines = []
    for k, v in test_config.get("env_vars", {}).items():
        env_setup_lines.append(f'os.environ["{k}"] = "{v}"')
    env_setup = "\n".join(env_setup_lines)
    
    # 테스트 스크립트 생성
    test_script = QUICK_TEST_SCRIPT.format(
        env_setup=env_setup,
        test_name=test_config["name"],
        workspace=WORKSPACE,
    )
    
    # 임시 스크립트 파일 생성
    script_path = f"{WORKSPACE}/_temp_component_test.py"
    with open(script_path, "w") as f:
        f.write(test_script)
    
    # 테스트 실행
    cmd = [
        "bash", "-c",
        f"source /home/conan/miniconda3/etc/profile.d/conda.sh && "
        f"conda activate llm_train && "
        f"cd {WORKSPACE} && "
        f"python {script_path}"
    ]
    
    start_time = time.time()
    try:
        process = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=TEST_CONFIG["timeout_seconds"]
        )
        
        result["duration_seconds"] = time.time() - start_time
        output = process.stdout + process.stderr
        
        # 결과 분석
        if "✅ TEST PASSED" in output:
            result["status"] = "PASSED"
        elif "ERROR_TYPE: tensor_size_mismatch_0_vs_2048" in output:
            result["status"] = "FAILED"
            result["error_type"] = "tensor_size_mismatch"
            result["error_message"] = "tensor a (0) vs tensor b (2048)"
        elif "ERROR_TYPE: cuda_oom" in output:
            result["status"] = "OOM"
            result["error_type"] = "cuda_oom"
            result["error_message"] = "CUDA out of memory"
        elif "❌ TEST FAILED" in output:
            result["status"] = "FAILED"
            # 에러 메시지 추출
            for line in output.split("\n"):
                if "TEST FAILED:" in line:
                    result["error_message"] = line.split("TEST FAILED:")[-1].strip()[:200]
                    break
        else:
            result["status"] = "FAILED"
            result["error_message"] = f"Unknown error (exit code: {process.returncode})"
            
    except subprocess.TimeoutExpired:
        result["status"] = "TIMEOUT"
        result["error_message"] = f"Test timed out after {TEST_CONFIG['timeout_seconds']} seconds"
        result["duration_seconds"] = TEST_CONFIG["timeout_seconds"]
    except Exception as e:
        result["status"] = "ERROR"
        result["error_message"] = str(e)
        result["duration_seconds"] = time.time() - start_time
    
    # 임시 파일 삭제
    try:
        os.remove(script_path)
    except:
        pass
    
    result["end_time"] = datetime.now().isoformat()
    return result


def generate_report(results: list) -> str:
    """테스트 결과 리포트 생성"""
    report = []
    report.append("# Seqorth 컴포넌트 점검 리포트")
    report.append(f"\n**생성 시간**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"\n**에러 타입**: `RuntimeError: The size of tensor a (0) must match the size of tensor b (2048)`")
    
    report.append("\n## 요약")
    
    passed = sum(1 for r in results if r["status"] == "PASSED")
    failed = sum(1 for r in results if r["status"] == "FAILED")
    other = len(results) - passed - failed
    
    report.append(f"\n| 항목 | 값 |")
    report.append(f"|------|-----|")
    report.append(f"| 총 테스트 | {len(results)} |")
    report.append(f"| 성공 | {passed} |")
    report.append(f"| 실패 | {failed} |")
    report.append(f"| 기타 | {other} |")
    
    report.append("\n## 상세 결과")
    report.append("\n| 컴포넌트 | 설명 | 상태 | 소요 시간 | 에러 |")
    report.append("|----------|------|------|----------|------|")
    
    for r in results:
        status_emoji = {
            "PASSED": "✅",
            "FAILED": "❌",
            "OOM": "💾",
            "TIMEOUT": "⏰",
            "ERROR": "🔥",
        }.get(r["status"], "❓")
        
        error_msg = r.get("error_message", "-") or "-"
        if len(error_msg) > 50:
            error_msg = error_msg[:50] + "..."
        
        report.append(f"| `{r['name']}` | {r['description']} | {status_emoji} {r['status']} | {r['duration_seconds']:.1f}s | {error_msg} |")
    
    report.append("\n## 분석")
    
    # 문제 컴포넌트 식별
    baseline_passed = any(r["name"] == "baseline_no_seqorth" and r["status"] == "PASSED" for r in results)
    full_seqorth_failed = any(r["name"] == "full_seqorth" and r["status"] == "FAILED" for r in results)
    
    if baseline_passed:
        report.append("\n### 발견 사항")
        report.append("- ✅ **Seqorth 없이 Qwen3 MoE는 정상 작동** (baseline 테스트 통과)")
        
        if full_seqorth_failed:
            report.append("- ❌ **Seqorth 전체 활성화 시 에러 발생**")
            
            # 문제 컴포넌트 식별
            report.append("\n### 문제 컴포넌트 식별")
            
            problem_found = False
            for r in results:
                if r["name"] not in ["baseline_no_seqorth", "full_seqorth"]:
                    if r["status"] == "PASSED":
                        report.append(f"- 🎯 **`{r['name']}`** 비활성화 시 정상 작동")
                        report.append(f"  - → **이 컴포넌트({r['description']})가 문제의 원인!**")
                        problem_found = True
                    elif r["status"] == "FAILED":
                        report.append(f"- ⚠️ `{r['name']}` 비활성화해도 여전히 에러 발생")
            
            if not problem_found:
                report.append("- ⚠️ 단일 컴포넌트 비활성화로는 문제 해결 불가")
                report.append("- → **복합적인 문제 또는 여러 컴포넌트 동시 비활성화 필요**")
    else:
        report.append("\n### ⚠️ 주의")
        report.append("- baseline 테스트(Seqorth 없이 Qwen3 MoE)도 실패")
        report.append("- 문제가 Seqorth가 아닌 다른 곳에 있을 수 있음")
    
    report.append("\n## 권장 사항")
    
    # 문제 컴포넌트에 대한 권장 사항
    for r in results:
        if r["name"] not in ["baseline_no_seqorth", "full_seqorth"] and r["status"] == "PASSED":
            report.append(f"\n### `{r['name']}` 수정 필요")
            report.append(f"- **문제 컴포넌트**: {r['description']}")
            report.append("- **해결 방안**:")
            report.append("  1. DeepSpeed ZeRO-3와 호환되도록 재설계")
            report.append("  2. backward 중 tensor shape 불일치 원인 분석")
            report.append("  3. 분산 파라미터 접근 방식 수정")
            break
    
    report.append("\n---")
    report.append(f"\n*이 리포트는 자동 생성되었습니다.*")
    
    return "\n".join(report)


def main():
    print("=" * 70)
    print("Seqorth 컴포넌트 자동 점검 시작")
    print("=" * 70)
    print(f"테스트 개수: {len(COMPONENT_TESTS)}")
    print(f"타임아웃: {TEST_CONFIG['timeout_seconds']}초")
    print("=" * 70)
    
    results = []
    
    for i, test in enumerate(COMPONENT_TESTS):
        print(f"\n{'='*70}")
        print(f"[{i+1}/{len(COMPONENT_TESTS)}] 테스트: {test['name']}")
        print(f"  설명: {test['description']}")
        print(f"  환경 변수: {test.get('env_vars', {})}")
        print(f"  시작 시간: {datetime.now().strftime('%H:%M:%S')}")
        print("-" * 70)
        
        result = run_test(test)
        results.append(result)
        
        status_emoji = {"PASSED": "✅", "FAILED": "❌", "OOM": "💾", "TIMEOUT": "⏰"}.get(result["status"], "❓")
        print(f"\n  결과: {status_emoji} {result['status']}")
        if result["error_message"]:
            print(f"  에러: {result['error_message'][:100]}")
        print(f"  소요 시간: {result['duration_seconds']:.1f}초")
        
        # 중간 결과 저장
        with open(TEST_CONFIG["report_path"] + ".json", "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    
    # 최종 리포트 생성
    report = generate_report(results)
    with open(TEST_CONFIG["report_path"], "w") as f:
        f.write(report)
    
    print("\n" + "=" * 70)
    print("점검 완료!")
    print(f"리포트 저장 위치: {TEST_CONFIG['report_path']}")
    print("=" * 70)
    
    # 리포트 출력
    print("\n" + report)


if __name__ == "__main__":
    main()
