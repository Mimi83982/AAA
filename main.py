#!/usr/bin/env python3
"""
SmartDiet – CLI recipe recommender.
Runs completely inside the `.venv` (Python 3.13) environment.
"""

from pathlib import Path
import sys

# ── allow `import utils …` etc. when script launched from project root
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"
sys.path.append(str(SRC_PATH))

# ── local imports (resolved after sys.path tweak)
from utils import data_loader
from fuzzy import rules
from engine import recommender


def main() -> None:
    # 1. user inputs (NO diet_type yet)
    user_profile = {
        "age": 28,
        "height": 160,  # cm
        "weight": 50,   # kg
        "activity_level": "Low",
        "satiety": 3,   # optional
    }

    # 2. load recipe dataframe
    recipes_df, _ = data_loader.load_data()

    # 3. compute BMI
    bmi = data_loader.compute_bmi(user_profile["weight"], user_profile["height"])

    # 4. fuzzy memberships (age, activity, bmi, satiety)
    rules.get_fuzzy_memberships(user_profile, bmi)

    # 5. fuzzy output dict → includes diet_type memberships
    fuzzy_outputs = rules.get_fuzzy_output()

    # 6. pick best diet type
    best_diet = max(fuzzy_outputs["diet_type"], key=fuzzy_outputs["diet_type"].get)
    user_profile["diet_type"] = best_diet
    print(f"\n✅ Recommended Diet Type: {best_diet.capitalize()}")

    # 7. recipe recommendation
    top_recipes = recommender.recommend_recipes(user_profile, fuzzy_outputs, recipes_df)

    # 8. show result
    print("\n🥗 Top Recipe Recommendations:")
    print(top_recipes[["name", "calories", "diet_type", "score"]])


if __name__ == "__main__":
    main()
