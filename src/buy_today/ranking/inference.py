"""Recommendations from a saved ranking model, without a history snapshot."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from buy_today.progress import stage
from buy_today.ranking.domain import Predictor
from buy_today.ranking.snapshots import digest


def recommend(model_dir: Path | str, user_id: str, k: object = 10) -> pd.DataFrame:
    """Return ordered products with one-based ranks for a known training user.

    Only manifest.json and model.joblib are read. The format and model checksum
    are checked before deserialization; predict validates user_id and k.

    Args:
        model_dir (Path | str): Saved model and manifest directory.
        user_id (str): Known training user.
        k (object, default=10): Positive integer recommendation cutoff.

    Returns:
        pd.DataFrame: Ordered user, rank and product columns.

    Raises:
        ValueError: Model version or checksum is invalid.
    """
    model_dir = Path(model_dir).resolve()
    with stage("inference.load", model_dir=model_dir):
        manifest = json.loads((model_dir / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("format_version") != 1:
            raise ValueError("Expected ranking model format_version=1")
        model_path = model_dir / "model.joblib"
        if digest(model_path) != manifest["model"]["sha256"]:
            raise ValueError("Ranking model SHA-256 mismatch")
        model: Predictor[NDArray[np.object_], str, object] = joblib.load(model_path)
    with stage("inference.predict", user_id=user_id, k=k):
        products = model.predict(user_id, k)
        return pd.DataFrame(
            {
                "user_id": user_id,
                "rank": range(1, len(products) + 1),
                "product_id": products,
            }
        )


def export_recommendations(
    model_dir: Path | str,
    user_id: str,
    output_path: Path | str,
    *,
    k: object = 10,
) -> Path:
    """Write a UTF-8 CSV and return its absolute path, propagating errors.

    The destination must be outside the saved model directory and must not
    alias any of its files. Missing destination parents are created.

    Args:
        model_dir (Path | str): Saved model directory.
        user_id (str): Known training user.
        output_path (Path | str): Destination CSV outside the model directory.
        k (object, default=10): Positive integer recommendation cutoff.

    Returns:
        Path: Absolute destination path.

    Raises:
        ValueError: Output overlaps the model snapshot.
    """
    model_dir, output_path = Path(model_dir).resolve(), Path(output_path).resolve()
    if output_path == model_dir or model_dir in output_path.parents:
        raise ValueError("Recommendation output must be outside the model directory")
    if output_path.exists() and any(
        path.is_file() and output_path.samefile(path) for path in model_dir.rglob("*")
    ):
        raise ValueError("Recommendation output must not overwrite model directory files")
    recommendations = recommend(model_dir, user_id, k)
    with stage("inference.export", output_path=output_path):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame.to_csv(recommendations, output_path, index=False, encoding="utf-8")
    return output_path
