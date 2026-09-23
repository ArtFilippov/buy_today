"""Model strategies return a complete snapshot; their update policy is local."""

import pandas as pd
from sklearn.base import BaseEstimator

from buy_today.clustering.contracts import TrainableDistance
from buy_today.clustering import domain

ModelSnapshot = domain.ModelSnapshot[BaseEstimator, BaseEstimator | TrainableDistance, pd.DataFrame]
