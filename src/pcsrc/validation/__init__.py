# ==============================================================================
# src/pcsrc/validation/__init__.py
#
# Validation package public API.
#
# Purpose:
#   - Expose validation utilities through a stable package-level interface.
#   - Keep imports lightweight via lazy attribute loading.
#   - Provide a single namespace for metrics, statistical summaries, simulation
#     validators, and reproducibility checks.
#
# Design notes:
#   - This module intentionally avoids importing heavy optional dependencies.
#   - Public objects are resolved lazily when accessed.
#   - Missing optional modules raise clear AttributeError messages.
#
# Typical usage:
#
#   from pcsrc.validation import ValidationResult, validate_experiment_output
#
#   result = validate_experiment_output(...)
# ==============================================================================

from __future__ import annotations

from importlib import import_module
from typing import Any


# ------------------------------------------------------------------------------
# Public lazy export map
# ------------------------------------------------------------------------------

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    # metrics.py
    "MetricDirection": ("pcsrc.validation.metrics", "MetricDirection"),
    "MetricRecord": ("pcsrc.validation.metrics", "MetricRecord"),
    "MetricSummary": ("pcsrc.validation.metrics", "MetricSummary"),
    "ValidationMetric": ("pcsrc.validation.metrics", "ValidationMetric"),
    "aggregate_metric_records": ("pcsrc.validation.metrics", "aggregate_metric_records"),
    "compare_metric_summaries": ("pcsrc.validation.metrics", "compare_metric_summaries"),
    "compute_basic_statistics": ("pcsrc.validation.metrics", "compute_basic_statistics"),
    "format_metric_table": ("pcsrc.validation.metrics", "format_metric_table"),
    "normalize_metric_value": ("pcsrc.validation.metrics", "normalize_metric_value"),

    # checker.py
    "CheckSeverity": ("pcsrc.validation.checker", "CheckSeverity"),
    "CheckStatus": ("pcsrc.validation.checker", "CheckStatus"),
    "ValidationCheck": ("pcsrc.validation.checker", "ValidationCheck"),
    "ValidationReport": ("pcsrc.validation.checker", "ValidationReport"),
    "ValidationResult": ("pcsrc.validation.checker", "ValidationResult"),
    "combine_reports": ("pcsrc.validation.checker", "combine_reports"),
    "validate_experiment_output": ("pcsrc.validation.checker", "validate_experiment_output"),
    "validate_numeric_series": ("pcsrc.validation.checker", "validate_numeric_series"),
    "validate_required_keys": ("pcsrc.validation.checker", "validate_required_keys"),

    # statistical_tests.py
    "HypothesisTestResult": ("pcsrc.validation.statistical_tests", "HypothesisTestResult"),
    "StatisticalTest": ("pcsrc.validation.statistical_tests", "StatisticalTest"),
    "bootstrap_confidence_interval": ("pcsrc.validation.statistical_tests", "bootstrap_confidence_interval"),
    "paired_difference_test": ("pcsrc.validation.statistical_tests", "paired_difference_test"),
    "permutation_test": ("pcsrc.validation.statistical_tests", "permutation_test"),

    # robustness.py
    "RobustnessScenario": ("pcsrc.validation.robustness", "RobustnessScenario"),
    "RobustnessSummary": ("pcsrc.validation.robustness", "RobustnessSummary"),
    "evaluate_robustness": ("pcsrc.validation.robustness", "evaluate_robustness"),
    "summarize_robustness": ("pcsrc.validation.robustness", "summarize_robustness"),

    # reproducibility.py
    "ReproducibilityCheck": ("pcsrc.validation.reproducibility", "ReproducibilityCheck"),
    "ReproducibilityReport": ("pcsrc.validation.reproducibility", "ReproducibilityReport"),
    "check_reproducibility_metadata": ("pcsrc.validation.reproducibility", "check_reproducibility_metadata"),
    "hash_artifact": ("pcsrc.validation.reproducibility", "hash_artifact"),
}


__all__ = tuple(sorted(_LAZY_IMPORTS))


# ------------------------------------------------------------------------------
# Lazy import protocol
# ------------------------------------------------------------------------------

def __getattr__(name: str) -> Any:
    """Resolve public validation attributes lazily."""

    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module 'pcsrc.validation' has no attribute {name!r}")

    module_name, attribute_name = _LAZY_IMPORTS[name]

    try:
        module = import_module(module_name)
    except ModuleNotFoundError as exc:
        raise AttributeError(
            f"Unable to import {name!r} from {module_name!r}. "
            "The corresponding validation module is unavailable."
        ) from exc

    try:
        value = getattr(module, attribute_name)
    except AttributeError as exc:
        raise AttributeError(
            f"Module {module_name!r} does not define expected attribute "
            f"{attribute_name!r}."
        ) from exc

    globals()[name] = value

    return value


def __dir__() -> list[str]:
    """Return sorted public attributes for interactive discovery."""

    return sorted(set(globals()) | set(__all__))


# ------------------------------------------------------------------------------
# Package metadata
# ------------------------------------------------------------------------------

PACKAGE_NAME = "pcsrc.validation"
PACKAGE_DESCRIPTION = "Validation utilities for pcsrc experiments and simulations."
