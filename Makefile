# Experimental Pipeline v2 — stage targets (see README). PY defaults to the project venv.
PY ?= .venv/bin/python
RUN ?= runs/smoke
PANEL ?= $(RUN)/panel.npz
MODELS ?= models.mock.yaml

.PHONY: install test test-all calibrate smoke data-cn data-us refs pools narrate parse verify drivers judges analysis tables roster

install:
	python3 -m venv .venv && $(PY) -m pip install -r requirements.txt && $(PY) -m pip install -e .

test:            ## fast suite (CI)
	$(PY) -m pytest -m "not slow and not network"

test-all:        ## includes the 500-formula executor agreement and the full calibration set
	$(PY) -m pytest -m "not network"

calibrate:       ## planted-claims calibration (§10.7)
	$(PY) -m verify calibrate --out runs/calibration

data-cn:
	$(PY) -m data rebuild cn --universe csi500
data-us:
	$(PY) -m data rebuild us
refs:
	$(PY) -m data refs us

roster:
	$(PY) -m narrate check-roster --models $(MODELS)

# ---------------------------------------------------------------- offline end-to-end smoke run (mock LLMs)
smoke:
	mkdir -p $(RUN)
	$(PY) -m data synthetic --out $(PANEL) --stocks 60 --days 700
	$(MAKE) pools narrate parse verify drivers judges analysis tables RUN=$(RUN) PANEL=$(PANEL) MODELS=$(MODELS) SCALE=0.1 K=1 FAST=--fast AUTHORS=$(MODELS)

SCALE ?= 1.0
K ?= 3
FAST ?=
AUTHORS ?=
pools:
	$(PY) -m pools build --panel $(PANEL) --out $(RUN) --scale $(SCALE) $(FAST) $(if $(AUTHORS),--authors $(AUTHORS))
narrate:
	$(PY) -m narrate run --formulas $(RUN)/formulas.jsonl --panel $(PANEL) --run-dir $(RUN) --models $(MODELS) --k $(K) $(FAST)
parse:
	$(PY) -m parse run --rationales $(RUN)/rationales.jsonl --out $(RUN)/claims.jsonl --models $(MODELS)
verify:
	$(PY) -m verify run --formulas $(RUN)/formulas.jsonl --claims $(RUN)/claims.jsonl --rationales $(RUN)/rationales.jsonl --panel $(PANEL) --out $(RUN)/verdicts.jsonl $(FAST)
drivers:
	$(PY) -m verify drivers --formulas $(RUN)/formulas.jsonl --panel $(PANEL) --out $(RUN)/drivers.jsonl $(FAST)
judges:
	$(PY) -m judges run --run-dir $(RUN) --panel $(PANEL) --models $(MODELS) --judges B1,B2,B4,B5 $(FAST)
analysis:
	$(PY) -m analysis run --run-dir $(RUN) --n-boot 200
tables:
	$(PY) -m paper tables --run-dir $(RUN)
