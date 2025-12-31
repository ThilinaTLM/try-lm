# =============================================================================
# TryLM Tutorial Book - Makefile
# =============================================================================

# Configuration
LATEX = pdflatex
LATEX_FLAGS = -interaction=nonstopmode -halt-on-error -output-directory=output
BIBER = biber

# Main document
MAIN = main
OUTPUT_DIR = output
PDF = $(OUTPUT_DIR)/$(MAIN).pdf

# Source files
CHAPTERS = $(wildcard chapters/*.tex)
APPENDICES = $(wildcard appendices/*.tex)
FIGURES = $(wildcard figures/*.tex)
SOURCES = $(MAIN).tex preamble.tex $(CHAPTERS) $(APPENDICES) $(FIGURES)

# =============================================================================
# Targets
# =============================================================================

.PHONY: all clean view watch help

# Default target: build the PDF
all: $(PDF)

# Build the PDF (run twice for TOC and cross-references)
$(PDF): $(SOURCES) | $(OUTPUT_DIR)
	@echo "=== First LaTeX pass ==="
	$(LATEX) $(LATEX_FLAGS) $(MAIN).tex
	@echo "=== Second LaTeX pass (for TOC and references) ==="
	$(LATEX) $(LATEX_FLAGS) $(MAIN).tex
	@echo "=== Build complete: $(PDF) ==="

# Create output directory if it doesn't exist
$(OUTPUT_DIR):
	mkdir -p $(OUTPUT_DIR)

# Build with bibliography (if using biblatex)
bib: $(SOURCES) | $(OUTPUT_DIR)
	@echo "=== First LaTeX pass ==="
	$(LATEX) $(LATEX_FLAGS) $(MAIN).tex
	@echo "=== Running Biber ==="
	$(BIBER) $(OUTPUT_DIR)/$(MAIN)
	@echo "=== Second LaTeX pass ==="
	$(LATEX) $(LATEX_FLAGS) $(MAIN).tex
	@echo "=== Third LaTeX pass ==="
	$(LATEX) $(LATEX_FLAGS) $(MAIN).tex
	@echo "=== Build complete: $(PDF) ==="

# Quick build (single pass, for rapid iteration)
quick: | $(OUTPUT_DIR)
	@echo "=== Quick build (single pass) ==="
	$(LATEX) $(LATEX_FLAGS) $(MAIN).tex

# Clean auxiliary files but keep PDF
clean:
	@echo "=== Cleaning auxiliary files ==="
	rm -f $(OUTPUT_DIR)/*.aux
	rm -f $(OUTPUT_DIR)/*.log
	rm -f $(OUTPUT_DIR)/*.out
	rm -f $(OUTPUT_DIR)/*.toc
	rm -f $(OUTPUT_DIR)/*.bbl
	rm -f $(OUTPUT_DIR)/*.bcf
	rm -f $(OUTPUT_DIR)/*.blg
	rm -f $(OUTPUT_DIR)/*.run.xml
	rm -f $(OUTPUT_DIR)/*.fdb_latexmk
	rm -f $(OUTPUT_DIR)/*.fls
	rm -f $(OUTPUT_DIR)/*.synctex.gz
	@echo "=== Done ==="

# Clean everything including PDF
distclean: clean
	@echo "=== Removing output directory ==="
	rm -rf $(OUTPUT_DIR)
	@echo "=== Done ==="

# View the PDF (platform-specific)
view: $(PDF)
	@if command -v xdg-open > /dev/null; then \
		xdg-open $(PDF); \
	elif command -v open > /dev/null; then \
		open $(PDF); \
	elif command -v evince > /dev/null; then \
		evince $(PDF); \
	else \
		echo "No PDF viewer found. Open $(PDF) manually."; \
	fi

# Watch for changes and rebuild (requires inotifywait on Linux)
watch:
	@echo "=== Watching for changes (Ctrl+C to stop) ==="
	@while true; do \
		$(MAKE) quick; \
		echo "=== Waiting for changes... ==="; \
		inotifywait -e modify -e create -e delete $(SOURCES) 2>/dev/null || sleep 5; \
	done

# Help
help:
	@echo "TryLM Tutorial Book - Build System"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  all       Build the complete PDF (default)"
	@echo "  quick     Quick single-pass build for rapid iteration"
	@echo "  bib       Build with bibliography processing"
	@echo "  clean     Remove auxiliary files"
	@echo "  distclean Remove all generated files including PDF"
	@echo "  view      Open the PDF in default viewer"
	@echo "  watch     Watch for changes and auto-rebuild"
	@echo "  help      Show this help message"
	@echo ""
	@echo "Output: $(PDF)"
