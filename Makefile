PYTHON := .venv/bin/python
PYTHONPATH := src
LATEST_SUBMISSION := $(shell ls -t submissions/*.csv 2>/dev/null | head -n 1)
UPLOADED_SUBMISSION := submissions/submission_phd_below075_20260522.csv
VERIFY_SUBMISSION_OUT ?= /tmp/dm114_verify_submission.csv
# Hardcoded to match the archived uploaded submission so `make phd-below075`
# output is byte-comparable with `make verify-submission` on any day.
PHD_SUBMISSION := submissions/submission_phd_below075_20260522.csv

.PHONY: venv test check-data eda baselines cv-fast cv train-fast train validate-latest phd-below075 verify-submission artifacts ablation clean

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

phd-below075:
	@echo "=== PhD Training Menu cached re-blend (data distribution -> menu -> final submission) ==="
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/analyze_data_distribution.py --force-synthesis --emit-menu
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/multi_blend_grid.py --menu reports/training_menu_v1.json --fixed --out $(PHD_SUBMISSION)
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/validate_submission.py $(PHD_SUBMISSION)
	@echo "=== Done. The submission above is a cached-prediction re-blend from retained artifacts. ==="

verify-submission:
	@test -f $(UPLOADED_SUBMISSION) || (echo "Missing $(UPLOADED_SUBMISSION)"; exit 1)
	@echo "=== Regenerating cached final submission to $(VERIFY_SUBMISSION_OUT) ==="
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/analyze_data_distribution.py --force-synthesis --emit-menu
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/multi_blend_grid.py --menu reports/training_menu_v1.json --fixed --out $(VERIFY_SUBMISSION_OUT)
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/validate_submission.py $(VERIFY_SUBMISSION_OUT)
	$(PYTHON) scripts/compare_submission.py $(VERIFY_SUBMISSION_OUT) $(UPLOADED_SUBMISSION) --abs-tol 1e-9 --ulp-tol 16

artifacts:
	$(PYTHON) scripts/write_artifacts_manifest.py

ablation:
	@echo "=== Regenerating lag-2215 and SSL OOF predictions (Phase 1 + Phase 2) ==="
	PYTHONPATH=$(PYTHONPATH) python3 scripts/regen_lag_2215_oof.py
	PYTHONPATH=$(PYTHONPATH) python3 scripts/regen_ssl_oof.py
	@echo "=== Building 9-row controlled ablation + cross-leg rho bootstrap ==="
	python3 scripts/build_ablation_9row.py
	python3 scripts/compute_cross_leg_rho.py
	@echo "=== Done. Outputs: reports/ablation_9row.{csv,md}, reports/cross_leg_rho_bootstrap.json ==="

clean:
	find . -path ./.venv -prune -o -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -path ./.venv -prune -o -type d -name .pytest_cache -prune -exec rm -rf {} +
