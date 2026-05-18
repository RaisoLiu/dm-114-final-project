PYTHON := .venv/bin/python
PYTHONPATH := src
LATEST_SUBMISSION := $(shell ls -t submissions/*.csv 2>/dev/null | head -n 1)

.PHONY: venv test check-data eda baselines cv-fast cv train-fast train validate-latest clean

venv:
	python3 -m venv .venv
	$(PYTHON) -m pip install -r requirements.txt pytest kaggle

test:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pytest -q

check-data:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/check_data.py

eda:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/eda.py

baselines:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/baselines.py

cv-fast:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/cross_validate.py --fast

cv:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/cross_validate.py --models lightgbm,hgb,extra --folds 3 --valid-weeks 52

train-fast:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/train_predict.py --fast

train:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/train_predict.py --models lightgbm,hgb,extra

validate-latest:
	@if [ -z "$(LATEST_SUBMISSION)" ]; then \
		echo "No submissions/*.csv file found."; \
		exit 1; \
	fi
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/validate_submission.py $(LATEST_SUBMISSION)

clean:
	find . -path ./.venv -prune -o -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -path ./.venv -prune -o -type d -name .pytest_cache -prune -exec rm -rf {} +

