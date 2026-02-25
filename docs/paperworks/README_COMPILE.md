# SPECTRA 논문 PDF 컴파일 가이드

## 빠른 시작

### 방법 1: 자동 컴파일 스크립트 (권장)

```bash
cd /home/conan/workspace/llm_training/docs/paperworks
./compile_pdf.sh
```

스크립트가 자동으로 다음을 시도합니다:
1. 로컬 pdflatex (설치되어 있는 경우)
2. Docker (sudo 권한 필요)
3. 실패 시 안내 메시지 표시

### 방법 2: 온라인 LaTeX 컴파일러 (가장 쉬움)

1. **Overleaf.com** 접속 (무료 계정 생성)
2. 새 프로젝트 생성 → "Upload Project"
3. 다음 파일들을 업로드:
   - `spectra_icml2026.tex`
   - `spectra_references.bib`
   - `icml2026/` 폴더 전체 (또는 필요한 .sty, .bst 파일들)
4. "Recompile" 버튼 클릭
5. PDF 다운로드

**장점**: 설치 불필요, 즉시 사용 가능, 실시간 미리보기

### 방법 3: 로컬 LaTeX 설치 후 컴파일

#### Ubuntu/Debian:
```bash
sudo apt-get update
sudo apt-get install texlive-full
```

#### macOS:
```bash
brew install --cask mactex
```

#### 컴파일:
```bash
cd /home/conan/workspace/llm_training/docs/paperworks
export TEXINPUTS=".:$(pwd)/icml2026:"
pdflatex spectra_icml2026.tex
bibtex spectra_icml2026
pdflatex spectra_icml2026.tex
pdflatex spectra_icml2026.tex
```

### 방법 4: Docker 사용 (sudo 권한 필요)

```bash
cd /home/conan/workspace/llm_training/docs/paperworks
sudo docker run --rm \
    -v "$(pwd):/workspace" \
    -w /workspace \
    texlive/texlive:latest \
    bash -c "
        export TEXINPUTS='.:/workspace/icml2026:'
        pdflatex -interaction=nonstopmode spectra_icml2026.tex
        bibtex spectra_icml2026
        pdflatex -interaction=nonstopmode spectra_icml2026.tex
        pdflatex -interaction=nonstopmode spectra_icml2026.tex
    "
```

## 필요한 파일

- `spectra_icml2026.tex` - 메인 논문 파일
- `spectra_references.bib` - 참고문헌
- `icml2026.sty` - ICML 스타일 파일 (이미 복사됨)
- `icml2026.bst` - 참고문헌 스타일 (이미 복사됨)
- 기타 .sty 파일들 (algorithm.sty, algorithmic.sty 등)

## 문제 해결

### 오류: "icml2026.sty not found"
- 해결: `export TEXINPUTS=".:$(pwd)/icml2026:"` 설정 확인
- 또는 스타일 파일들을 현재 디렉토리로 복사 (이미 완료됨)

### 오류: "Bibliography not found"
- 해결: `bibtex spectra_icml2026` 실행 후 다시 `pdflatex` 실행

### 오류: "Undefined references"
- 해결: `pdflatex`를 2-3번 더 실행 (참조 해결을 위해)

## 출력 파일

컴파일 성공 시 생성되는 파일:
- `spectra_icml2026.pdf` - 최종 PDF (이것을 확인하세요!)
- `spectra_icml2026.aux` - 보조 파일
- `spectra_icml2026.bbl` - 참고문헌 파일
- `spectra_icml2026.blg` - BibTeX 로그
- `spectra_icml2026.log` - 컴파일 로그

## 추천 방법

**가장 빠른 방법**: Overleaf.com 사용 (설치 불필요, 즉시 사용 가능)
