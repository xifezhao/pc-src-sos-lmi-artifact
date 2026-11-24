# Artifact Evaluation Readiness Checklist (Candidate Status)

> **Important:** ACM/IEEE Artifact Evaluation badges are awarded solely by an independent Artifact Evaluation Committee (AEC) or journal reviewers—never self-awarded by authors. All items below reflect **author-side preparation status** prior to independent review.

| Badge Category | Current Status | Evidence / Remaining Requirement |
|---|---|---|
| **Artifacts Available** | Candidate (Pending Archival DOI) | Public GitHub repository (`v2.1.0`) with MIT license; permanent Zenodo DOI to be minted upon manuscript acceptance. |
| **Artifacts Evaluated – Functional** | Candidate (CI Verified) | GitHub Actions CI passes SHA-256 verification and `pytest -v tests/`; unified CLI `scripts/reproduce_paper.py --fast` / `--full` runs end-to-end without commercial solvers. |
| **Artifacts Evaluated – Reusable** | Candidate (Prepared) | Modular `src/pcsrc/` package, pinned `requirements-lock.txt`, real Dual-Certificate-verified LUT export, complete NGSIM provenance/CC BY-SA citation; awaits independent AEC usability review. |
| **Results Reproduced** | Pending External Review | Author-side full reproduction verified on Google Colab CPU (`636.9 s`); badge requires confirmation by an independent third-party evaluation team. |
