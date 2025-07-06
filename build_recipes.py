# build_recipes.py

import pandas as pd
import ast
import numpy as np
import os

# ────────────────────────────────────────────────────────────────
# 1. Detect and load source CSV
# ────────────────────────────────────────────────────────────────
if os.path.exists("data/RAW_recipes.csv"):
    src = "../Fuzzy_Project_Diet copy/data/RAW_recipes.csv"
    df = pd.read_csv(src, usecols=["name", "ingredients", "nutrition"])

    df["nutrition"] = df["nutrition"].apply(lambda x: x if isinstance(x, str) else "[]")
    df["calories"] = df["nutrition"].apply(
        lambda x: ast.literal_eval(x)[0] if isinstance(x, str) and x.startswith("[") else np.nan
    )

elif os.path.exists("data/PP_recipes.csv"):
    src = "../Fuzzy_Project_Diet copy/data/PP_recipes.csv"
    df = pd.read_csv(src)

    # Check required columns
    if "calories" not in df.columns:
        raise ValueError("PP_recipes.csv is missing 'calories' column.")

    ingredient_cols = [c for c in df.columns if c.startswith("ingredient_")]
    df["ingredients"] = df[ingredient_cols].astype(str).agg(", ".join, axis=1)

    if "nutrition" not in df.columns:
        # Generate dummy nutrition if missing
        df["nutrition"] = df["calories"].apply(lambda x: [x] + [0]*6)
    else:
        df["nutrition"] = df["nutrition"].apply(lambda x: x if isinstance(x, str) else "[]")

else:
    raise FileNotFoundError("No source file found. Expected either 'RAW_recipes.csv' or 'PP_recipes.csv' in /data")

print(f"📦 Loaded {len(df):,} rows from {src}")

# ────────────────────────────────────────────────────────────────
# 2. Clean data
# ────────────────────────────────────────────────────────────────
df = df.dropna(subset=["name", "ingredients", "calories"])
df = df[df["calories"] > 0]

# ────────────────────────────────────────────────────────────────
# 3. Tag diet type
# ────────────────────────────────────────────────────────────────
def tag(row):
    ingr = row["ingredients"].lower()
    cals = row["calories"]
    if any(w in ingr for w in ["tempeh", "tofu", "lentil", "beans", "chickpea"]):
        return "vegan"
    if cals < 350 and any(w in ingr for w in ["salad", "greens", "lettuce", "cauliflower"]):
        return "low_carb"
    if any(w in ingr for w in ["chicken", "turkey", "beef", "fish"]):
        return "high_protein"
    return "balanced"

df["diet_type"] = df.apply(tag, axis=1)
df = df[df["diet_type"].isin(["vegan", "balanced", "high_protein", "low_carb"])]

# ────────────────────────────────────────────────────────────────
# 4. Sample 200 (evenly) and add prep_time + meal_type
# ────────────────────────────────────────────────────────────────
target = 200
df = (
    df.groupby("diet_type", group_keys=False)
    .apply(lambda g: g.sample(min(target // 4, len(g)), random_state=42))
    .reset_index(drop=True)
)

df["prep_time"] = np.random.randint(5, 31, size=len(df))

def assign_meal_type(minutes):
    if minutes <= 10:
        return "breakfast"
    elif minutes <= 20:
        return "lunch"
    else:
        return "dinner"

df["meal_type"] = df["prep_time"].apply(assign_meal_type)

# ────────────────────────────────────────────────────────────────
# 5. Final formatting
# ────────────────────────────────────────────────────────────────
df.insert(0, "recipe_id", range(1, len(df) + 1))
df = df[["recipe_id", "name", "nutrition", "ingredients", "calories", "diet_type", "prep_time", "meal_type"]]

df.to_csv("data/recipes.csv", index=False)
print(f"✅ Wrote {len(df)} recipes to data/recipes.csv")
