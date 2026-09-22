"""Replaceable estimators exposing fit(X) and pairwise(X, Y=None)."""

from buy_today.clustering.distances.temporal_distances import TimestampDistance

__all__ = ["TimestampDistance"]
