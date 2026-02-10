PY ?= .venv/bin/python
PIP ?= .venv/bin/pip
SHELL := /bin/bash

VENV := .venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

CONFIG ?= configs/smoke.yaml

.PHONY: setup data train eval report all clean

setup:
	@# venv bootstrap: host may lack ensurepip and system pip may be PEP668-managed
	@if [ -d .venv ] && [ ! -x .venv/bin/python ]; then rm -rf .venv; fi
	@if [ ! -d .venv ]; then python3 -m venv --without-pip .venv; fi
	@if [ ! -x .venv/bin/pip ]; then python3 -c "import pathlib,urllib.request; p=pathlib.Path('.venv/get-pip.py'); p.parent.mkdir(parents=True,exist_ok=True); urllib.request.urlretrieve('https://bootstrap.pypa.io/get-pip.py', p)"; .venv/bin/python .venv/get-pip.py; fi
	@scripts/bootstrap_venv.sh

data: setup
	@$(PY) -m librispeech_mrm.cli --config $(CONFIG) data

train: setup
	@$(PY) -m librispeech_mrm.cli --config $(CONFIG) train

eval: setup
	@$(PY) -m librispeech_mrm.cli --config $(CONFIG) eval

report: setup
	@$(PY) -m librispeech_mrm.cli --config $(CONFIG) report

all: setup data train eval report

clean:
	@rm -rf artifacts/runs artifacts/results.json artifacts/report.md
