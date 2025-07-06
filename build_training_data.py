"""
Build training_dataset.csv from 20 k Food.com recipes + interactions.

Features
────────
age, gender, bmi, activity, satiety
calories, total_fat, sugar, sodium, protein, saturated_fat, carbs
diet_type  (vegan / balanced / high_protein / low_carb)
satisfaction  (1–5 star average per recipe)

Outputs
───────
data/training_dataset.csv   (~8 k rows)
"""
from pathlib import Path
import pandas as pd, numpy as np

DATA = Path("data")
RNG  = np.random.default_rng(42)

# 1. load
recipes = pd.read_csv(DATA / "RAW_recipes.csv",
                      usecols=["id","nutrition","tags"])
reviews = pd.read_csv(DATA / "RAW_interactions.csv",
                      usecols=["recipe_id","rating"])

# 2. nutrition → 7 cols
nut_cols = ["calories","total_fat","sugar","sodium",
            "protein","saturated_fat","carbs"]
recipes[nut_cols] = (
    recipes["nutrition"]
    .str.strip("[]").str.split(",", expand=True)
    .astype(float)
)
recipes = recipes.drop(columns="nutrition")

# 3. diet_type from tags
def diet_from_tags(t: str) -> str|None:
    t = t.lower()
    if "vegan" in t:                                        return "vegan"
    if "vegetarian" in t:                                   return "balanced"
    if "keto" in t or "low-carb" in t or "low carb" in t:   return "low_carb"
    if "high-protein" in t or "paleo" in t:                 return "high_protein"
    return None

recipes["diet_type"] = recipes["tags"].fillna("").apply(diet_from_tags)
recipes = recipes.dropna(subset=["diet_type"]).drop(columns="tags")

# 3b. one-hot flags for each diet class (used as features)
for d in ["vegan", "balanced", "high_protein", "low_carb"]:
    recipes[f"is_{d}"] = (recipes["diet_type"] == d).astype(int)

# 4. satisfaction from interactions
avg_rating = reviews.groupby("recipe_id")["rating"].mean().round(1)
recipes = recipes.join(avg_rating, on="id")
recipes = recipes.rename(columns={"rating":"satisfaction"}).dropna()

# 5. synthetic user profile (1 row per recipe)
N = len(recipes)
recipes["gender"]         = RNG.choice(["M","F"], size=N)
recipes["age"]            = RNG.integers(18,70, size=N)
recipes["height"]         = RNG.integers(150,190, size=N)
recipes["weight"]         = recipes["height"]*0.45 + RNG.integers(-20,25,size=N)
recipes["activity_level"] = RNG.choice(["Low","Medium","High"], size=N)
recipes["satiety"]        = RNG.integers(1,6, size=N)

# 6. reorder & save
diet_flags = ["is_vegan", "is_balanced", "is_high_protein", "is_low_carb"]
out_cols = ["age","gender","height","weight","activity_level",
            "satiety","diet_type","satisfaction"] + nut_cols + diet_flags
out = recipes[out_cols]
out.to_csv(DATA / "user_feedback.csv", index=False)
print(f"✅  Wrote {len(out)} rows to data/user_feedback.csv")
