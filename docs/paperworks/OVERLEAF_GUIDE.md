# Overleaf에서 PDF 컴파일하기 (가장 쉬운 방법!)

## 단계별 가이드

### 1단계: Overleaf 접속
- https://www.overleaf.com 접속
- 무료 계정 생성 (Google/GitHub 로그인 가능)

### 2단계: 새 프로젝트 생성
- "New Project" → "Upload Project" 클릭
- 또는 "Blank Project" 생성 후 파일 업로드

### 3단계: 파일 업로드

다음 파일들을 업로드하세요:

**필수 파일:**
- `seqorth_icml2026.tex` (메인 논문)
- `seqorth_references.bib` (참고문헌)

**스타일 파일들 (icml2026 폴더에서):**
- `icml2026.sty`
- `icml2026.bst`
- `algorithm.sty`
- `algorithmic.sty`
- `fancyhdr.sty`

**또는 간단하게:**
- `icml2026/` 폴더 전체를 zip으로 압축해서 업로드

### 4단계: 컴파일
- Overleaf에서 "Recompile" 버튼 클릭
- PDF가 자동으로 생성됩니다!

### 5단계: PDF 다운로드
- 생성된 PDF 우클릭 → "Download PDF"

## 장점
- ✅ 설치 불필요
- ✅ 즉시 사용 가능
- ✅ 실시간 미리보기
- ✅ 협업 기능
- ✅ 버전 관리

## 문제 해결

**오류: "icml2026.sty not found"**
- 해결: icml2026 폴더의 모든 .sty 파일을 프로젝트 루트에 업로드

**오류: "Bibliography not found"**
- 해결: Compiler를 "pdfLaTeX"로 설정하고 "Recompile" 클릭

**참고문헌이 표시되지 않음**
- 해결: Compiler 메뉴에서 "pdfLaTeX" → "BibTeX" → "pdfLaTeX" → "pdfLaTeX" 순서로 실행
