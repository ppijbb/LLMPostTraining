#!/bin/bash
# SPECTRA 컴포넌트별 자동 점검 - 백그라운드 실행용
# 결과는 spectra_component_report.md에 저장됩니다.

WORKSPACE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPORT_FILE="${WORKSPACE}/spectra_component_report.md"
LOG_DIR="${WORKSPACE}/component_test_logs"
MASTER_LOG="${WORKSPACE}/component_test_master.log"

exec > "$MASTER_LOG" 2>&1

mkdir -p "$LOG_DIR"

source /home/conan/miniconda3/etc/profile.d/conda.sh
conda activate llm_train

echo "========================================================================"
echo "SPECTRA 컴포넌트 자동 점검 시작"
echo "시작 시간: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================================================"

# 결과 배열
declare -A RESULTS
declare -A ERRORS
declare -A DURATIONS

# 테스트 함수
run_single_test() {
    local name="$1"
    local desc="$2"
    local env_var="$3"
    
    echo ""
    echo "========================================================================"
    echo "[테스트] $name"
    echo "설명: $desc"
    echo "환경변수: $env_var"
    echo "시작: $(date '+%H:%M:%S')"
    echo "------------------------------------------------------------------------"
    
    local log_file="${LOG_DIR}/${name}.log"
    local start_time=$(date +%s)
    
    # 환경 변수 설정
    export SPECTRA_TEST_MODE=1
    export SPECTRA_TEST_MAX_STEPS=1
    [ -n "$env_var" ] && export $env_var
    
    # 실행 (타임아웃 45분)
    cd "$WORKSPACE"
    timeout 2700 bash training/spectra_sft/run_spectra.sh > "$log_file" 2>&1
    local exit_code=$?
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    # 환경 변수 정리
    unset SPECTRA_TEST_MODE SPECTRA_TEST_MAX_STEPS
    [ -n "$env_var" ] && unset "${env_var%%=*}"
    
    # 결과 분석
    local status="UNKNOWN"
    local error=""
    
    if grep -q "size of tensor a (0)" "$log_file" 2>/dev/null; then
        status="FAILED"
        error="tensor_size_mismatch_0_vs_2048"
    elif grep -q "CUDA out of memory" "$log_file" 2>/dev/null; then
        status="OOM"
        error="CUDA_OOM"
    elif grep -q "\[TEST MODE\] ✅ Step" "$log_file" 2>/dev/null; then
        status="PASSED"
    elif [ $exit_code -eq 124 ]; then
        status="TIMEOUT"
        error="timeout_45min"
    else
        status="FAILED"
        error=$(grep -E "RuntimeError|Error:" "$log_file" 2>/dev/null | head -1 | cut -c1-80)
    fi
    
    RESULTS[$name]="$status"
    ERRORS[$name]="$error"
    DURATIONS[$name]="$duration"
    
    echo "결과: $status"
    echo "에러: $error"
    echo "소요시간: ${duration}초"
    echo "종료: $(date '+%H:%M:%S')"
}

# 테스트 실행
echo ""
echo "테스트 1/7: baseline_no_spectra"
run_single_test "baseline_no_spectra" "SPECTRA 완전 비활성화" "SPECTRA_DISABLE_ALL=1"

echo ""
echo "테스트 2/7: disable_expert_dispatch"
run_single_test "disable_expert_dispatch" "Expert dispatch 비활성화" "SPECTRA_DISABLE_EXPERT_DISPATCH=1"

echo ""
echo "테스트 3/7: disable_router"
run_single_test "disable_router" "SPECTRARouter 비활성화" "SPECTRA_DISABLE_ROUTER=1"

echo ""
echo "테스트 4/7: disable_shared_experts"
run_single_test "disable_shared_experts" "shared_experts 비활성화" "SPECTRA_DISABLE_SHARED_EXPERTS=1"

echo ""
echo "테스트 5/7: disable_intent_gated"
run_single_test "disable_intent_gated" "IntentGatedContextCell 비활성화" "SPECTRA_DISABLE_INTENT_GATED=1"

echo ""
echo "테스트 6/7: disable_expression_proj"
run_single_test "disable_expression_proj" "ExpressionProjector 비활성화" "SPECTRA_DISABLE_EXPRESSION_PROJ=1"

echo ""
echo "테스트 7/7: full_spectra"
run_single_test "full_spectra" "SPECTRA 전체 활성화" ""

# 리포트 생성
echo ""
echo "========================================================================"
echo "리포트 생성 중..."
echo "========================================================================"

cat > "$REPORT_FILE" << 'REPORT_HEADER'
# SPECTRA 컴포넌트 점검 리포트

REPORT_HEADER

echo "**생성 시간**: $(date '+%Y-%m-%d %H:%M:%S')" >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"
echo "**에러 타입**: \`RuntimeError: The size of tensor a (0) must match the size of tensor b (2048)\`" >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"

# 요약
passed=0
failed=0
for name in baseline_no_spectra disable_expert_dispatch disable_router disable_shared_experts disable_intent_gated disable_expression_proj full_spectra; do
    [ "${RESULTS[$name]}" == "PASSED" ] && ((passed++))
    [ "${RESULTS[$name]}" == "FAILED" ] && ((failed++))
done

echo "## 요약" >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"
echo "| 항목 | 값 |" >> "$REPORT_FILE"
echo "|------|-----|" >> "$REPORT_FILE"
echo "| 총 테스트 | 7 |" >> "$REPORT_FILE"
echo "| 성공 | $passed |" >> "$REPORT_FILE"
echo "| 실패 | $failed |" >> "$REPORT_FILE"
echo "| 기타 | $((7-passed-failed)) |" >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"

# 상세 결과
echo "## 상세 결과" >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"
echo "| 컴포넌트 | 설명 | 상태 | 소요시간 | 에러 |" >> "$REPORT_FILE"
echo "|----------|------|------|----------|------|" >> "$REPORT_FILE"

declare -A DESCS
DESCS[baseline_no_spectra]="SPECTRA 완전 비활성화"
DESCS[disable_expert_dispatch]="Expert dispatch 비활성화"
DESCS[disable_router]="SPECTRARouter 비활성화"
DESCS[disable_shared_experts]="shared_experts 비활성화"
DESCS[disable_intent_gated]="IntentGatedContextCell 비활성화"
DESCS[disable_expression_proj]="ExpressionProjector 비활성화"
DESCS[full_spectra]="SPECTRA 전체 활성화"

for name in baseline_no_spectra disable_expert_dispatch disable_router disable_shared_experts disable_intent_gated disable_expression_proj full_spectra; do
    status="${RESULTS[$name]}"
    error="${ERRORS[$name]}"
    duration="${DURATIONS[$name]}"
    desc="${DESCS[$name]}"
    
    case $status in
        "PASSED") status_str="✅ PASSED" ;;
        "FAILED") status_str="❌ FAILED" ;;
        "OOM") status_str="💾 OOM" ;;
        "TIMEOUT") status_str="⏰ TIMEOUT" ;;
        *) status_str="❓ $status" ;;
    esac
    
    echo "| \`$name\` | $desc | $status_str | ${duration}s | ${error:-"-"} |" >> "$REPORT_FILE"
done

echo "" >> "$REPORT_FILE"

# 분석
echo "## 분석" >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"

if [ "${RESULTS[baseline_no_spectra]}" == "PASSED" ]; then
    echo "### 발견 사항" >> "$REPORT_FILE"
    echo "- ✅ **SPECTRA 없이 Qwen3 MoE는 정상 작동**" >> "$REPORT_FILE"
    
    if [ "${RESULTS[full_spectra]}" == "FAILED" ]; then
        echo "- ❌ **SPECTRA 전체 활성화 시 에러 발생**" >> "$REPORT_FILE"
        echo "" >> "$REPORT_FILE"
        echo "### 문제 컴포넌트 식별" >> "$REPORT_FILE"
        
        for name in disable_expert_dispatch disable_router disable_shared_experts disable_intent_gated disable_expression_proj; do
            if [ "${RESULTS[$name]}" == "PASSED" ]; then
                echo "- 🎯 **\`$name\`** 비활성화 시 정상 → **${DESCS[$name]}가 문제!**" >> "$REPORT_FILE"
            fi
        done
    fi
else
    echo "### ⚠️ 주의" >> "$REPORT_FILE"
    echo "- baseline 테스트도 실패 → 문제가 SPECTRA 외부에 있을 수 있음" >> "$REPORT_FILE"
fi

echo "" >> "$REPORT_FILE"
echo "## 권장 사항" >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"

for name in disable_expert_dispatch disable_router disable_shared_experts disable_intent_gated disable_expression_proj; do
    if [ "${RESULTS[$name]}" == "PASSED" ] && [ "${RESULTS[full_spectra]}" == "FAILED" ]; then
        echo "### \`$name\` 수정 필요" >> "$REPORT_FILE"
        echo "- **문제 컴포넌트**: ${DESCS[$name]}" >> "$REPORT_FILE"
        echo "- **해결 방안**:" >> "$REPORT_FILE"
        echo "  1. DeepSpeed ZeRO-3와 호환되도록 재설계" >> "$REPORT_FILE"
        echo "  2. backward 중 tensor shape 불일치 원인 분석" >> "$REPORT_FILE"
        break
    fi
done

echo "" >> "$REPORT_FILE"
echo "---" >> "$REPORT_FILE"
echo "*자동 생성된 리포트*" >> "$REPORT_FILE"

echo ""
echo "========================================================================"
echo "점검 완료!"
echo "종료 시간: $(date '+%Y-%m-%d %H:%M:%S')"
echo "리포트: $REPORT_FILE"
echo "========================================================================"

# 완료 알림 파일 생성
touch "${WORKSPACE}/component_test_DONE"
