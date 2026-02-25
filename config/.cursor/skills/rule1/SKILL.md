---
name: research
description: This is a new research skill
---

# Overview

[ 연구 목표 ]
- [ ] CV < 0.001 달성: IntentGatedContext, Orthogonality Specialized Experts, Load Balancing Bias 강화.
- [ ] cv < 0.001 수치 달성. maxvio 0.01 달성. expert speciality 확보. 
- [ ] Top-tier 논문 연구로 진행하는 작업이며 어떠한 베낌도 허용되지 않음.

모든 작업시 반드시 다음 수칙을 적용합니다.

[ 작업시 제한사항 ]
- [ ] DeepSpeed ZeRO-3 Stage 3 유지: 파라미터 수동 수집/in-place 연산 금지.
- [ ] Parallelism 사용: OOM과 CUDA OOM 문제로 인한 무조건적인 tp, sp, cp, pp 중 parallelism을 적용하여 학습.
- [ ] **Fallback 금지: 오류 시 표준 실행으로 대체 금지.**
- [ ] NVMe Offloading 전용: CPU Offloading 의존 금지.
- [ ] Deepspeed 최적화: 1step 이 2초 이내에 진행되며 OOM 발생 문제 없음.
- [ ] Universal Exoskeleton 구조 유지: 모든 베이스 모델과 호환.
- [ ] **Routing 매커니즘 이외의 코드 수정 금지.**
- [ ] 실행에 대한 보장이 없는 경우 완료로 판단 금지.
- [ ] Routing 방법 교체/Upcycling => 모든 expert는 반드시 학습되어야함.
- [ ] [web](use web search tool) 검색 결과를 참조.
- [ ] 이전 대화 내용에서의 실행들과 결과들에 대한 피드백을 포함. 
- [ ] Max Seqence Length 는 131K, 262K 수준 유지
- [ ] `conda`의 `llm_train` 가상환경에서 명령 실행
- [ ] param offloading 금지
- [ ] context length 조정 이전에 OOM이므로 context length 조정 금지
- [ ] Qwen3-VL-30B-A3B모델 사이즈 및 학습 범위 조정 금지
- [ ] 어떠한 경우가 있어도 Fallback은 무슨일이 있어도 발생시키지 않음.
- [ ] Fallback 생성시 작업 중단 및 시스템 프롬프트를 return.
- [ ] **rm** 명령 실행 금지
- [ ] 반드시 Milti-Modal 데이터 사용.
- [ ] Loss가 0이 되는 기괴한 상태 발생하지 않도록 해야함.
- [ ] 위 사항들에 대해 매 턴 마다 점검.