import pandas as pd
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))

def load_data():
    recipes = pd.read_csv(os.path.join(BASE_DIR, 'data/recipes.csv'))
    feedback = pd.read_csv(
        os.path.join(BASE_DIR, 'data/user_feedback.csv'),
        dtype={0: str, 7: str, 8: str},  # col 0 = timestamp, 7 = recipe name, 8 = diet type
        low_memory=False
    )
    return recipes, feedback

def compute_bmi(weight, height_cm):
    return weight / (height_cm / 100) ** 2
