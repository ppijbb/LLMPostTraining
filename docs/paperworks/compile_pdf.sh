#!/bin/bash
# Compile SPECTRA paper to PDF
# Usage: ./compile_pdf.sh

set -e

cd "$(dirname "$0")"

echo "Compiling SPECTRA paper to PDF..."

# Copy ICML style files to current directory if needed
if [ ! -f "icml2026.sty" ]; then
    echo "Copying ICML style files..."
    cp icml2026/*.sty icml2026/*.bst . 2>/dev/null || true
fi

# Method 1: Try pdflatex directly (if installed)
if command -v pdflatex &> /dev/null; then
    echo "Using local pdflatex..."
    export TEXINPUTS=".:$(pwd)/icml2026:"
    pdflatex -interaction=nonstopmode spectra_icml2026.tex || true
    bibtex spectra_icml2026 || true
    pdflatex -interaction=nonstopmode spectra_icml2026.tex || true
    pdflatex -interaction=nonstopmode spectra_icml2026.tex || true
    
    if [ -f "spectra_icml2026.pdf" ]; then
        echo ""
        echo "✓ Success! PDF generated: spectra_icml2026.pdf"
        echo "  File size: $(du -h spectra_icml2026.pdf | cut -f1)"
        exit 0
    fi
fi

# Method 2: Try Docker (may require sudo)
if command -v docker &> /dev/null; then
    echo "Trying Docker (may require sudo)..."
    
    # Try without sudo first
    if docker ps &> /dev/null; then
        DOCKER_CMD="docker"
    elif sudo docker ps &> /dev/null 2>&1; then
        DOCKER_CMD="sudo docker"
    else
        echo "Docker not accessible. Trying other methods..."
        DOCKER_CMD=""
    fi
    
    if [ -n "$DOCKER_CMD" ]; then
        $DOCKER_CMD run --rm \
            -v "$(pwd):/workspace" \
            -w /workspace \
            texlive/texlive:latest \
            bash -c "
                export TEXINPUTS='.:/workspace/icml2026:'
                pdflatex -interaction=nonstopmode spectra_icml2026.tex || true
                bibtex spectra_icml2026 || true
                pdflatex -interaction=nonstopmode spectra_icml2026.tex || true
                pdflatex -interaction=nonstopmode spectra_icml2026.tex || true
            "
        
        if [ -f "spectra_icml2026.pdf" ]; then
            echo ""
            echo "✓ Success! PDF generated: spectra_icml2026.pdf"
            echo "  File size: $(du -h spectra_icml2026.pdf | cut -f1)"
            exit 0
        fi
    fi
fi

# Method 3: Provide instructions
echo ""
echo "✗ Could not compile PDF automatically."
echo ""
echo "Please choose one of the following options:"
echo ""
echo "Option 1: Install LaTeX locally"
echo "  Ubuntu/Debian: sudo apt-get install texlive-full"
echo "  macOS: brew install --cask mactex"
echo "  Then run: pdflatex spectra_icml2026.tex"
echo ""
echo "Option 2: Use Docker with sudo"
echo "  sudo ./compile_pdf.sh"
echo ""
echo "Option 3: Use online LaTeX compiler"
echo "  - Upload spectra_icml2026.tex and spectra_references.bib to Overleaf.com"
echo "  - Upload icml2026/ folder as well"
echo "  - Compile online"
echo ""
echo "Option 4: Manual compilation (if LaTeX is installed)"
echo "  export TEXINPUTS='.:$(pwd)/icml2026:'"
echo "  pdflatex spectra_icml2026.tex"
echo "  bibtex spectra_icml2026"
echo "  pdflatex spectra_icml2026.tex"
echo "  pdflatex spectra_icml2026.tex"
echo ""
exit 1
