# Artifact Evaluation Reviewer Script

```bash
pip install -e . && pip install numpy scipy cvxpy clarabel scs pyyaml matplotlib pytest
python scripts/run_paper_experiments_full.py
pytest tests/ -v
```
