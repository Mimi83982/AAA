#!/usr/bin/env python3
"""
Local ANFIS satisfaction-inference helpers.
"""

from pathlib import Path
import numpy as np
import torch
import joblib

# -------- load model + scaler once --------
BASE = Path(__file__).resolve().parents[2]
SCALER = joblib.load(BASE / "models" / "scaler_satisfaction.pkl")

from models.anfis_diet import AnfisNet

NET = AnfisNet(input_dim=11, output_dim=5)
NET.load_state_dict(torch.load(BASE / "models" / "anfis_satisfaction.pth", map_location="cpu"))
NET.eval()

# -------- public api --------
def score_vectors(vectors: list[list[float]]) -> list[float]:
    """
    vectors : list of 11‑element feature lists.
    returns : list of floats 0‒1 preference score.
    """
    if not vectors:
        return []
    Xs = SCALER.transform(np.array(vectors, dtype=np.float32))
    with torch.no_grad():
        logits = NET(torch.tensor(Xs))
        probs  = torch.softmax(logits, dim=1)
        exp    = torch.arange(5, dtype=torch.float32)
        pref   = (probs * exp).sum(dim=1) / 4.0   # map 0‑4 → 0‑1
    return pref.tolist()

def infer_single(vec: list[float]) -> float:
    """Convenience wrapper for a single 11‑feature vector."""
    return score_vectors([vec])[0]
