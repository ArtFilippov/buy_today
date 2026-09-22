"""Executable reports for the prepared Olist dataset."""

from buy_today.auto_eda.deda import report_drift
from buy_today.auto_eda.drift import DriftThresholds
from buy_today.auto_eda.eda import report_dataset

__all__ = ["DriftThresholds", "report_dataset", "report_drift"]
