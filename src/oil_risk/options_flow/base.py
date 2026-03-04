from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date

import pandas as pd


class OptionsFlowProvider(ABC):
    @abstractmethod
    def fetch_daily_metrics(self, ticker: str, dt: date) -> pd.DataFrame:
        """Return rows with metric_name, metric_value for one ticker/day."""
