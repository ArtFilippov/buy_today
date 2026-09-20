"""Executable reports for the prepared Olist dataset."""

from prak.auto_eda.deda import report_drift
from prak.auto_eda.drift import DriftThresholds
from prak.auto_eda.eda import report_dataset

__all__ = ["DriftThresholds", "report_dataset", "report_drift"]
