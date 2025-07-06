from __future__ import annotations
import re
from typing import Dict, Any, List

import numpy as np
import pandas as pd

from utils.data_loader import compute_bmi
from anfis_local.infer import score_vectors

# ───────── Weights (tuned) ─────────
W_FUZZY_DIET = 1.5     # ↓ less dominant
W_CALORIE    = 3.0
W_QUICK      = 1.0
W_ANFIS_PREF = 3.5     # ↑ gives more variation

# ───────── Helpers ─────────
_NUM_EXTRACT = re.compile(r"([-+]?\d*\.?\d+)")

def _to_float_series(s: pd.Series) -> pd.Series:
    cleaned = (
        s.astype(str)
         .str.replace(",", "", regex=False)
         .str.extract(_NUM_EXTRACT, expand=False)
    )
    return pd.to_numeric(cleaned, errors="coerce").fillna(0.0)

def _calorie_bonus(cals: float, bmi: float) -> float:
    if np.isnan(cals) or np.isnan(bmi): return 0.0
    if bmi < 18.5:   return min(cals / 1000.0, 1.0)
    if bmi > 25:     return max(0.0, (600.0 - cals) / 600.0)
    return max(0.0, 1.0 - abs(cals - 450.0) / 450.0)

def _feature_vector(p: Dict[str, Any], row: pd.Series) -> list[float]:
    bmi = compute_bmi(p["weight"], p["height"])
    gender   = 1 if p.get("gender", "M") == "M" else 0
    activity = {"Low": 0, "Medium": 1, "High": 2}.get(p["activity_level"], 1)
    return [
        p["age"], gender, bmi, activity,
        row["calories"], row["total_fat"], row["sugar"], row["sodium"],
        row["protein"], row["saturated_fat"], row["carbs"],
    ]

def recommend_recipes(user_profile: Dict[str, Any],
                      fuzzy_out: Dict[str, Dict[str, float]],
                      recipes_df: pd.DataFrame,
                      feedback_df=None,
                      top_n: int = 3) -> pd.DataFrame:
    df = recipes_df.copy()

    df = df.rename(columns={"calories_kcal": "calories", "fat_total": "total_fat"})
    EXPECTED = ["calories","total_fat","sugar","sodium",
                "protein","saturated_fat","carbs","prep_time"]
    for col in EXPECTED:
        if col not in df.columns: df[col] = 0.0
        df[col] = _to_float_series(df[col])

    bmi_val = compute_bmi(user_profile["weight"], user_profile["height"])

    vecs, rows = [], []
    for _, r in df.iterrows():
        vecs.append(_feature_vector(user_profile, r)); rows.append(r)
    if not rows: return pd.DataFrame(columns=df.columns.tolist()+["score"])

    pref = score_vectors(vecs)       # 0‑1

    scores=[]
    for i,r in enumerate(rows):
        diet_key=str(r["diet_type"]).strip().lower()
        s  = fuzzy_out["diet_type"].get(diet_key,0.0)*W_FUZZY_DIET
        s += _calorie_bonus(r["calories"], bmi_val) *W_CALORIE
        s += (1.0 if r["prep_time"]<=15 else 0.0)   *W_QUICK
        s += pref[i]                                *W_ANFIS_PREF
        scores.append(s)

    df["score"]=np.round(scores,3)
    return df.sort_values("score",ascending=False).head(top_n).reset_index(drop=True)

def plan_day(profile: Dict[str,Any], fuzzy_out: Dict[str,Dict[str,float]],
             recipes_df: pd.DataFrame, per_session:int=3)->pd.DataFrame:
    rows=[]
    for meal in ("breakfast","lunch","dinner"):
        sub=recipes_df[recipes_df["meal_type"].str.lower()==meal]
        best=recommend_recipes(profile,fuzzy_out,sub,top_n=per_session)
        if not best.empty:
            best.loc[:,"meal_type"]=meal.title()
            rows.extend(best.to_dict(orient="records"))
    return pd.DataFrame(rows)
