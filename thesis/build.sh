#!/bin/bash

# LaTeX Build Script with Full Paths
# This script avoids ENOENT errors by using absolute paths

# Set paths
LATEXMK="/Library/TeX/texbin/latexmk"
PDFLATEX="/Library/TeX/texbin/pdflatex"
BIBER="/Library/TeX/texbin/biber"

# Create output directories
mkdir -p output tmp

echo "Building LaTeX document..."

# Option 1: Use latexmk (recommended)
if [ "$1" = "latexmk" ] || [ -z "$1" ]; then
    echo "Using latexmk..."
    $LATEXMK -pdf -synctex=1 --shell-escape -auxdir=tmp -outdir=output tese.tex
fi

# Option 2: Manual build sequence
if [ "$1" = "manual" ]; then
    echo "Using manual build sequence..."
    
    # First run
    $PDFLATEX --aux-directory=tmp -output-directory=output tese.tex
    
    # Run Biber
    $BIBER output/tese
    
    # Second run
    $PDFLATEX --aux-directory=tmp -output-directory=output tese.tex
    
    # Third run
    $PDFLATEX --aux-directory=tmp -output-directory=output tese.tex
fi

echo "Build complete! Check output/tese.pdf"
