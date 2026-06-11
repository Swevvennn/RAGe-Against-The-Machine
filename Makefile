VENV := /tmp/venv
PY := PYTHONPATH=/home/mosmond/RAG/src $(VENV)/bin/python3
PIP := $(VENV)/bin/pip

install:
	python3 -m venv $(VENV)
	$(PIP) install -r requirements.txt

run:
	$(PY) -m src.student

debug:
	$(PY) -m pdb -m src/student

lint:
	$(PY) flake8 src/
	$(PY) mypy src/ --warn-return-any --warn-unused-ignores \
		--ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	$(PY) flake8 src/
	$(PY) mypy src/ --strict

clean:
	rm -rf __pycache__
	rm -rf */__pycache__
	rm -rf __pycache__

fclean: clean
	rm -rf /tmp/venv
