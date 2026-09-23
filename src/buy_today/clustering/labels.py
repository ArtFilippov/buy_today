"""Validation of portable keyed cluster assignments, independent of training."""

import pandas as pd

from buy_today.schema import ROW_KEY


def check_labels(labels: pd.DataFrame) -> None:
    """Validate the portable keyed assignment table, including integer IDs.

    Args:
        labels (pd.DataFrame): Portable keyed cluster assignments.

    Raises:
        ValueError: Schema, keys or assignment values are invalid.
    """
    if tuple(labels.columns) != (*ROW_KEY, "cluster_id"):
        raise ValueError(f"labels must contain exactly {(*ROW_KEY, 'cluster_id')}")
    if labels.empty or labels.isna().any().any():
        raise ValueError("labels must be nonempty and contain no missing values")
    if labels.duplicated(list(ROW_KEY)).any():
        raise ValueError("Duplicate row keys in labels")
    for column in ("order_item_id", "cluster_id"):
        if not pd.api.types.is_integer_dtype(labels[column].dtype):
            raise ValueError(f"{column} in labels must contain integers")
    if not labels["order_id"].map(lambda value: isinstance(value, str)).all():
        raise ValueError("order_id in labels must contain strings")
