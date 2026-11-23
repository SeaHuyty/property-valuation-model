"""Load a trained price MLP and predict prices.

    from engines.neural_network.inference import PricePredictor
    p = PricePredictor("models/neural_network/model.pt")
    p.predict([{"property_type": "Condo", "district": "Chamkar Mon", "area_m2": 60, ...}])

Missing feature keys get neutral defaults (unknown category, 0 for counts/flags);
missing area_m2 / bedrooms / bathrooms also set the matching *_missing flag.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch

from engines.common import LAB
from engines.neural_network.networks import PriceMLP
from engines.neural_network.preprocessing import Preprocessor


class PricePredictor:
    def __init__(self, path="models/neural_network/model.pt"):
        path = Path(path)
        ckpt = torch.load(path if path.is_absolute() else LAB / path, map_location="cpu", weights_only=False)
        self.pre = Preprocessor.from_state(ckpt["preprocessor"])
        self.model = PriceMLP(**ckpt["model_args"])
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()

    def _frame(self, rows) -> pd.DataFrame:
        df = pd.DataFrame(rows)
        for col in self.pre.categorical:
            if col not in df:
                df[col] = "Unknown"
        for col in self.pre.numeric + self.pre.binary:
            missing_flag = f"{col}_missing"
            if col not in df:
                df[col] = 0.0
                if missing_flag in self.pre.binary and missing_flag not in df:
                    df[missing_flag] = 1
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        return df

    @torch.no_grad()
    def predict(self, rows) -> list[float]:
        """rows: list of dicts (or a DataFrame) with the training feature columns. Returns USD prices."""
        df = rows if isinstance(rows, pd.DataFrame) else self._frame(rows)
        cat, num = self.pre.transform(df)
        log_price = self.model(torch.from_numpy(cat), torch.from_numpy(num)).numpy()
        return self.pre.inverse_target(log_price).tolist()
