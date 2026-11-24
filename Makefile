.PHONY: install test reproduce-fast reproduce-all verify-hashes clean

install:
	pip install -e .[dev]

test:
	PYTHONPATH=src pytest -v tests/

reproduce-fast:
	PYTHONPATH=src python3 scripts/reproduce_paper.py --fast

reproduce-all:
	PYTHONPATH=src python3 scripts/reproduce_paper.py --full
	PYTHONPATH=src pytest -v tests/

verify-hashes:
	shasum -a 256 -c data/reference/golden_hashes.sha256

clean:
	rm -rf .pytest_cache *.egg-info results/reproduced
	find . -type d -name __pycache__ -exec rm -rf {} +
