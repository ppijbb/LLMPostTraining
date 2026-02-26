#!/bin/bash
# Seqorth 컴포넌트별 자동 점검 스크립트
# 각 컴포넌트를 비활성화하면서 backward 에러 발생 여부를 확인합니다.

set -e

WORKSPACE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPORT_FILE="${WORKSPACE}/seqorth_component_report.md"
JSON_FILE="${WORKSPACE}/seqorth_component_report.json"
LOG_DIR="${WORKSPACE}/component_test_logs"

# 로그 디렉토리 생성
mkdir -p "$LOG_DIR"

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Conda 환경 활성화
source /home/conan/miniconda3/etc/profile.d/conda.sh
conda activate llm_train

# 테스트 함수
run_test() {
    local test_name="$1"
    local description="$2"
    shift 2
    local env_vars=("$@")
    
    local log_file="${LOG_DIR}/${test_name}.log"
    local start_time=$(date +%s)
    
    echo "========================================================================"
    echo "테스트: $test_name"
    echo "설명: $description"
    echo "환경 변수: ${env_vars[*]}"
    echo "시작 시간: $(date '+%H:%M:%S')"
    echo "------------------------------------------------------------------------"
    
    # 환경 변수 설정
    export SEQORTH_TEST_MODE=1
    export SEQORTH_TEST_MAX_STEPS=1
    for env_var in "${env_vars[@]}"; do
        export "$env_var"
    done
    
    # 학습 실행 (타임아웃 15분)
    cd "$WORKSPACE"
    timeout 900 bash training/seqorth_sft/run_seqorth.sh > "$log_file" 2>&1
    local exit_code=$?
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    # 환경 변수 초기화
    unset SEQORTH_TEST_MODE
    unset SEQORTH_TEST_MAX_STEPS
    for env_var in "${env_vars[@]}"; do
        local var_name="${env_var%%=*}"
        unset "$var_name"
    done
    
    # 결과 분석
    local status="UNKNOWN"
    local error_msg=""
    
    if grep -q "size of tensor a (0)" "$log_file" 2>/dev/null; then
        status="FAILED"
        error_msg="tensor size mismatch (0 vs 2048)"
    elif grep -q "CUDA out of memory" "$log_file" 2>/dev/null; then
        status="OOM"
        error_msg="CUDA OOM"
    elif grep -q "Training step" "$log_file" 2>/dev/null || [ $exit_code -eq 0 ]; then
        # 첫 번째 스텝이 완료되었거나 정상 종료
        if grep -q "0%|" "$log_file" 2>/dev/null; then
            # 아직 0%라면 스텝 완료 전에 종료된 것
            if grep -q "RuntimeError" "$log_file" 2>/dev/null; then
                status="FAILED"
                error_msg=$(grep "RuntimeError" "$log_file" | head -1 | cut -c1-100)
            else
                status="PASSED"
            fi
        else
            status="PASSED"
        fi
    elif [ $exit_code -eq 124 ]; then
        status="TIMEOUT"
        error_msg="Timeout (15min)"
    else
        status="FAILED"
        error_msg=$(grep -E "Error|Exception" "$log_file" | head -1 | cut -c1-100)
    fi
    
    # 결과 출력
    case $status in
        "PASSED") echo -e "${GREEN}✅ PASSED${NC}" ;;
        "FAILED") echo -e "${RED}❌ FAILED: $error_msg${NC}" ;;
        "OOM") echo -e "${YELLOW}💾 OOM${NC}" ;;
        "TIMEOUT") echo -e "${YELLOW}⏰ TIMEOUT${NC}" ;;
        *) echo -e "❓ UNKNOWN" ;;
    esac
    echo "소요 시간: ${duration}초"
    echo "로그 파일: $log_file"
    echo ""
    
    # JSON 결과 저장 (append)
    echo "{\"name\":\"$test_name\",\"description\":\"$description\",\"status\":\"$status\",\"error\":\"$error_msg\",\"duration\":$duration}" >> "${JSON_FILE}.tmp"
    
    # 결과 반환
    echo "$status"
}

# 리포트 생성 함수
generate_report() {
    echo "# Seqorth 컴포넌트 점검 리포트" > "$REPORT_FILE"
    echo "" >> "$REPORT_FILE"
    echo "**생성 시간**: $(date '+%Y-%m-%d %H:%M:%S')" >> "$REPORT_FILE"
    echo "" >> "$REPORT_FILE"
    echo "## 상세 결과" >> "$REPORT_FILE"
    echo "" >> "$REPORT_FILE"
    echo "| 컴포넌트 | 설명 | 상태 | 소요 시간 | 에러 |" >> "$REPORT_FILE"
    echo "|----------|------|------|----------|------|" >> "$REPORT_FILE"
    
    while IFS= read -r line; do
        name=$(echo "$line" | jq -r '.name')
        desc=$(echo "$line" | jq -r '.description')
        status=$(echo "$line" | jq -r '.status')
        error=$(echo "$line" | jq -r '.error' | cut -c1-50)
        duration=$(echo "$line" | jq -r '.duration')
        
        case $status in
            "PASSED") status_str="✅ PASSED" ;;
            "FAILED") status_str="❌ FAILED" ;;
            "OOM") status_str="💾 OOM" ;;
            "TIMEOUT") status_str="⏰ TIMEOUT" ;;
            *) status_str="❓ $status" ;;
        esac
        
        echo "| \`$name\` | $desc | $status_str | ${duration}s | $error |" >> "$REPORT_FILE"
    done < "${JSON_FILE}.tmp"
    
    # JSON 파일로 변환
    echo "[" > "$JSON_FILE"
    sed '$ ! s/$/,/' "${JSON_FILE}.tmp" >> "$JSON_FILE"
    echo "]" >> "$JSON_FILE"
    rm -f "${JSON_FILE}.tmp"
    
    echo "" >> "$REPORT_FILE"
    echo "## 분석" >> "$REPORT_FILE"
    echo "" >> "$REPORT_FILE"
    
    # 분석 추가
    if grep -q '"status":"PASSED".*baseline_no_seqorth' "$JSON_FILE" 2>/dev/null; then
        echo "- ✅ Seqorth 없이 Qwen3 MoE는 정상 작동" >> "$REPORT_FILE"
    fi
    
    if grep -q '"status":"PASSED"' "$JSON_FILE" 2>/dev/null; then
        echo "" >> "$REPORT_FILE"
        echo "### 성공한 테스트 (문제 컴포넌트 식별)" >> "$REPORT_FILE"
        grep '"status":"PASSED"' "$JSON_FILE" | while IFS= read -r line; do
            name=$(echo "$line" | jq -r '.name')
            desc=$(echo "$line" | jq -r '.description')
            echo "- 🎯 **\`$name\`** 비활성화 시 정상 → **$desc가 문제!**" >> "$REPORT_FILE"
        done
    fi
    
    echo "" >> "$REPORT_FILE"
    echo "---" >> "$REPORT_FILE"
    echo "*이 리포트는 자동 생성되었습니다.*" >> "$REPORT_FILE"
}

# 메인 실행
echo "========================================================================"
echo "Seqorth 컴포넌트 자동 점검 시작"
echo "========================================================================"
echo ""

# 임시 JSON 파일 초기화
> "${JSON_FILE}.tmp"

# 테스트 실행
TESTS=(
    "baseline_no_seqorth|Seqorth 완전 비활성화 (Qwen3 원래 MoE)|SEQORTH_DISABLE_ALL=1"
    "disable_expert_dispatch|Expert dispatch 비활성화|SEQORTH_DISABLE_EXPERT_DISPATCH=1"
    "disable_router|SeqorthRouter 비활성화|SEQORTH_DISABLE_ROUTER=1"
    "disable_shared_experts|shared_experts 비활성화|SEQORTH_DISABLE_SHARED_EXPERTS=1"
    "disable_intent_gated|IntentGatedContextCell 비활성화|SEQORTH_DISABLE_INTENT_GATED=1"
    "disable_expression_proj|ExpressionProjector 비활성화|SEQORTH_DISABLE_EXPRESSION_PROJ=1"
    "full_seqorth|Seqorth 전체 활성화|"
)

total=${#TESTS[@]}
current=0

for test_def in "${TESTS[@]}"; do
    IFS='|' read -r name desc env_var <<< "$test_def"
    ((current++))
    echo "[$current/$total]"
    
    if [ -n "$env_var" ]; then
        run_test "$name" "$desc" "$env_var"
    else
        run_test "$name" "$desc"
    fi
done

# 리포트 생성
echo "========================================================================"
echo "리포트 생성 중..."
generate_report

echo "========================================================================"
echo "점검 완료!"
echo "리포트: $REPORT_FILE"
echo "JSON: $JSON_FILE"
echo "========================================================================"

# 리포트 출력
cat "$REPORT_FILE"
