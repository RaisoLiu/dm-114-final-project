# Makefile for DM 114 Final Project Report (Group 3)
# Produces DM_project_Group_3.pdf from LaTeX + generated figures
# Reproducible: fetches IEEEtran if missing, generates figures from repo artifacts

SHELL := /bin/bash
PYTHON := python3
ROOT := $(shell pwd)/..
REPORT_ROOT := $(shell pwd)
PDF := DM_project_Group_3.pdf
TEX := DM_project_Group_3.tex
TECTONIC_VERSION := 0.16.9
TECTONIC := $(ROOT)/.tools/tectonic-$(TECTONIC_VERSION)/tectonic
TECTONIC_ARCHIVE := $(ROOT)/.tools/tectonic-$(TECTONIC_VERSION)-aarch64-unknown-linux-musl.tar.gz
TECTONIC_URL := https://github.com/tectonic-typesetting/tectonic/releases/download/tectonic%40$(TECTONIC_VERSION)/tectonic-$(TECTONIC_VERSION)-aarch64-unknown-linux-musl.tar.gz

.PHONY: all report figures clean distclean check

all: report

# Main target
report: figures $(PDF)
	@echo "=== Report built: $(PDF) ==="
	@pdfinfo $(PDF) | grep -E 'Pages|Title|Creator' || true
	@echo "Run 'make check' for content verification."

$(PDF): $(TEX) figures ieee/IEEEtran.cls
	@echo "=== Compiling LaTeX with IEEEtran from ./ieee/ ==="
	@if command -v pdflatex >/dev/null 2>&1; then \
		echo "Using pdflatex"; \
		TEXINPUTS=".:ieee:" pdflatex -interaction=nonstopmode -halt-on-error $(TEX) > /tmp/latex1.log 2>&1 || (cat /tmp/latex1.log | tail -40; exit 1); \
		TEXINPUTS=".:ieee:" pdflatex -interaction=nonstopmode -halt-on-error $(TEX) > /tmp/latex2.log 2>&1 || (cat /tmp/latex2.log | tail -40; exit 1); \
	elif [ -x "$(TECTONIC)" ]; then \
		echo "Using bundled tectonic"; \
		"$(TECTONIC)" --keep-logs --keep-intermediates $(TEX) > /tmp/tectonic.log 2>&1 || (cat /tmp/tectonic.log | tail -80; exit 1); \
	else \
		echo "pdflatex missing; fetching user-space tectonic $(TECTONIC_VERSION)"; \
		$(MAKE) --no-print-directory "$(TECTONIC)" || exit 1; \
		"$(TECTONIC)" --keep-logs --keep-intermediates $(TEX) > /tmp/tectonic.log 2>&1 || (cat /tmp/tectonic.log | tail -80; exit 1); \
	fi
	@echo "=== LaTeX compile successful ==="

$(TECTONIC):
	@mkdir -p "$(ROOT)/.tools/tectonic-$(TECTONIC_VERSION)"
	@if [ ! -f "$(TECTONIC_ARCHIVE)" ]; then \
		echo "Downloading tectonic $(TECTONIC_VERSION) for linux-aarch64..."; \
		curl -fsSL "$(TECTONIC_URL)" -o "$(TECTONIC_ARCHIVE)"; \
	fi
	@tar -xzf "$(TECTONIC_ARCHIVE)" -C "$(ROOT)/.tools/tectonic-$(TECTONIC_VERSION)"
	@chmod +x "$(TECTONIC)"

# Ensure IEEEtran is present (download if needed for full reproducibility)
ieee/IEEEtran.cls:
	@mkdir -p ieee
	@if [ ! -f ieee/IEEEtran.cls ]; then \
		echo "Fetching IEEEtran.cls for offline reproducibility..."; \
		curl -fsSL -o ieee/IEEEtran.cls https://raw.githubusercontent.com/bryant1410/IEEEtran/master/IEEEtran.cls || \
		(echo "WARNING: Could not download IEEEtran.cls. Place it manually in report/ieee/"; exit 1); \
	fi

# Generate all guiding figures (idempotent, uses only repo artifacts + optional on-demand experiments per plan §2.6)
figures:
	@echo "=== Generating/refreshing guiding figures ==="
	@mkdir -p figures
	$(PYTHON) figures/generate_figures.py || (echo "Figure generation script missing or failed - will create stub in next step"; exit 1)
	@echo "=== Figures ready ==="

# Content & reproducibility checks (run after build)
check:
	@echo "=== PDF content & spec checks ==="
	@test -f $(PDF) || (echo "PDF not found"; exit 1)
	@pdfinfo $(PDF) | grep -q "Pages" && echo "  - PDF exists and has page info"
	@pdftotext $(PDF) - | grep -q "Team 3" && echo "  - Contains 'Team 3'"
	@pdftotext $(PDF) - | grep -q "0.7628" && echo "  - Contains final score 0.7628"
	@pdftotext $(PDF) - | grep -q "below075_20260522.csv" && echo "  - Contains final submission filename"
	@pdftotext $(PDF) - | grep -q "github.com/RaisoLiu/dm-114-final-project" && echo "  - Contains canonical GitHub link"
	@! pdftotext $(PDF) - | grep -q "dm-114-final-project-claude" || (echo "Old GitHub URL still present"; exit 1)
	@! pdftotext $(PDF) - | grep -qi "legitimate" || (echo "Integrity-sensitive word 'legitimate' still present"; exit 1)
	@! pdftotext $(PDF) - | grep -q "region-resampled" || (echo "Incorrect bootstrap wording still present"; exit 1)
	@echo "  - All key content checks passed (manual visual QA of figures still required)"
	@echo "=== To fully verify reproducibility of the 0.7628 submission: ==="
	@echo "  cd .. && make verify-submission"

clean:
	rm -f *.aux *.log *.out *.toc *.bbl *.blg *.synctex.gz
	rm -f $(PDF)

distclean: clean
	rm -rf figures/ ieee/
	rm -f $(PDF)

# Quick smoke: just check that the LaTeX would find the class (no full compile)
smoke:
	@if command -v pdflatex >/dev/null 2>&1; then echo "pdflatex OK"; elif [ -x "$(TECTONIC)" ]; then echo "bundled tectonic OK"; else echo "LaTeX engine missing (make will fetch tectonic if network is available)"; fi
	@test -f ieee/IEEEtran.cls && echo "IEEEtran.cls present" || echo "IEEEtran.cls missing (run make)"
	@echo "Smoke OK"
