import sys
import os
import skfuzzy as fuzz
import pandas as pd

feedback_df = pd.read_csv("../data/user_feedback.csv")

# Path config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from fuzzy.rules import diet_simulator
from engine.recommender import recommend_recipes
from utils.data_loader import load_data
from fuzzy import membership  # To access fuzzy sets


def interpret_diet_type(score):
    if score <= 0.25:
        return 'Vegan'
    elif score <= 0.5:
        return 'Balanced'
    elif score <= 0.75:
        return 'High Protein'
    else:
        return 'Low Carb'


# Match fuzzy label to recipe dataset label
label_map = {
    'Vegan': 'Vegan',
    'Balanced': 'Vegetarian',
    'High Protein': 'Omnivore',
    'Low Carb': 'Keto'
}

# Step 1: Simulate user input
user_profile = {
    'age': 65,
    'activity_level': 'Low',
    'bmi': 27,
    'satiety_level': 'Medium',
    'diet_type': label_map['Low Carb']  # from fuzzy output
}

# Step 2: Map activity & satiety to numeric values
activity_map = {'low': 2, 'medium': 5, 'high': 8}
satiety_map = {'low': 1, 'medium': 3, 'high': 5}

# Step 3: Run fuzzy control system
simulator = diet_simulator
simulator.input['age'] = user_profile['age']
simulator.input['activity'] = activity_map[user_profile['activity_level'].lower()]
simulator.input['bmi'] = user_profile['bmi']
simulator.input['satiety'] = satiety_map[user_profile['satiety_level'].lower()]
simulator.compute()

# Step 4: Print fuzzy result (diet type score)
fuzzy_score = simulator.output['diet_type']
print("Fuzzy Diet Type Output (score):", fuzzy_score)

# Step 5: Generate fuzzy memberships
bmi_sets = membership.generate_bmi_fuzzy_sets()
activity_sets = membership.generate_activity_fuzzy_sets()
age_sets = membership.generate_age_fuzzy_sets()

fuzzy_outputs = {
    'bmi': {
        'underweight': fuzz.interp_membership(bmi_sets['x'], bmi_sets['underweight'], user_profile['bmi']),
        'normal': fuzz.interp_membership(bmi_sets['x'], bmi_sets['normal'], user_profile['bmi']),
        'overweight': fuzz.interp_membership(bmi_sets['x'], bmi_sets['overweight'], user_profile['bmi']),
        'obese': fuzz.interp_membership(bmi_sets['x'], bmi_sets['obese'], user_profile['bmi'])
    },
    'activity': {
        'low': fuzz.interp_membership(activity_sets['x'], activity_sets['low'], activity_map[user_profile['activity_level'].lower()]),
        'medium': fuzz.interp_membership(activity_sets['x'], activity_sets['medium'], activity_map[user_profile['activity_level'].lower()]),
        'high': fuzz.interp_membership(activity_sets['x'], activity_sets['high'], activity_map[user_profile['activity_level'].lower()])
    },
    'age': {
        'young': fuzz.interp_membership(age_sets['x'], age_sets['young'], user_profile['age']),
        'adult': fuzz.interp_membership(age_sets['x'], age_sets['adult'], user_profile['age']),
        'elderly': fuzz.interp_membership(age_sets['x'], age_sets['elderly'], user_profile['age'])
    }
}

# Step 6: Load data
recipes_df, _ = load_data()

# Step 7: Recommend recipes
recommendations = recommend_recipes(user_profile, fuzzy_outputs, recipes_df, feedback_df, top_n=3)

# Step 8: Show results
print("\nTop Recommended Recipes:")
print(recommendations[['name', 'diet_type', 'prep_time', 'calories', 'score']])
