# 빠른 PDF 생성 가이드

## 🚀 가장 빠른 방법: Overleaf 사용 (5분)

### 1. Overleaf 접속
https://www.overleaf.com → 무료 계정 생성

### 2. 프로젝트 업로드
- "New Project" → "Upload Project"
- `seqorth_overleaf.tar.gz` 파일 업로드 (또는 아래 파일들 직접 업로드)

**필요한 파일:**
```
seqorth_icml2026.tex          (메인 논문)
seqorth_references.bib        (참고문헌)
icml2026/                     (스타일 파일 폴더)
  ├── icml2026.sty
  ├── icml2026.bst
  ├── algorithm.sty
  ├── algorithmic.sty
  └── fancyhdr.sty
```

### 3. 컴파일
- "Recompile" 버튼 클릭
- PDF 자동 생성!

---

## 💻 로컬 컴파일 (LaTeX 설치 필요)

### LaTeX 설치
```bash
# Ubuntu/Debian
sudo apt-get install texlive-full

# macOS  
brew install --cask mactex
```

### 컴파일
```bash
cd /home/conan/workspace/llm_training/docs/paperworks
export TEXINPUTS=".:$(pwd)/icml2026:"
pdflatex seqorth_icml2026.tex
bibtex seqorth_icml2026
pdflatex seqorth_icml2026.tex
pdflatex seqorth_icml2026.tex
```

생성된 `seqorth_icml2026.pdf` 파일을 확인하세요!

---

## 📦 Docker 사용 (sudo 필요)

```bash
cd /home/conan/workspace/llm_training/docs/paperworks
sudo docker run --rm \
    -v "$(pwd):/workspace" \
    -w /workspace \
    texlive/texlive:latest \
    bash -c "
        export TEXINPUTS='.:/workspace/icml2026:'
        pdflatex -interaction=nonstopmode seqorth_icml2026.tex
        bibtex seqorth_icml2026
        pdflatex -interaction=nonstopmode seqorth_icml2026.tex
        pdflatex -interaction=nonstopmode seqorth_icml2026.tex
    "
```

---

## ✅ 추천: Overleaf 사용

**이유:**
- 설치 불필요
- 즉시 사용 가능
- 실시간 미리보기
- 협업 기능
- 버전 관리

**단계:**
1. Overleaf.com 접속
2. `seqorth_overleaf.tar.gz` 업로드
3. "Recompile" 클릭
4. PDF 다운로드

완료! 🎉
