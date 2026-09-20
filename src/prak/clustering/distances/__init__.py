"""Replaceable estimators exposing fit(X) and pairwise(X, Y=None)."""

from prak.clustering.distances.temporal_distances import TimestampDistance

__all__ = ["TimestampDistance"]
