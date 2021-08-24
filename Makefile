.PHONY: init fresh install test-gpu
python_bin ?= python
init:
	$(info **************** Installing Dependencies ****************)
	$(python_bin) -m pip install -r requirements.txt
	$(python_bin) -m pip install --no-deps tf-models-official

install:
	$(info **************** Installing Pip Package ****************)
	$(python_bin) -m pip install .
	$(python_bin) -m pip install --no-deps tf-models-official
fresh:

	$(info **************** Creating Virutal Environment ****************)
	-rm -rf venv
	$(python_bin) -m venv venv
ifeq ($(OS),Windows_NT)
	@echo Windows Detected
	python_bin="venv\Scripts\python.exe" make # Windows
else
	@echo Linux/MacOS Detected
	python_bin="venv/bin/python" make # Linux
endif



test-gpu:
	$(info **************** Running GPU Test ****************)
	$(python_bin) tests/tensorflow-gpu.py
