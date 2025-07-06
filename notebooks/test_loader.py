import sys
import os

# Add src/ to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from utils.data_loader import load_data, compute_bmi

recipes, feedback = load_data()

print("Recipes sample:")
print(recipes.head())

print("\nFeedback sample:")
print(feedback.head())

print("\nSample BMI:", compute_bmi(weight=60, height_cm=165))
