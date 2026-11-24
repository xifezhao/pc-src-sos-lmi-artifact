.PHONY: install test verify fast full clean

install:
	python3 -m pip install -r requirements-lock.txt
	python3 -m pip install -e .

verify:
	shasum -a 256 -c data/reference/golden_hashes.sha256

test:
	PYTHONPATH=src pytest -v tests/

fast:
	PYTHONPATH=src python3 scripts/reproduce_paper.py --fast

full:
	PYTHONPATH=src python3 scripts/reproduce_paper.py --full
