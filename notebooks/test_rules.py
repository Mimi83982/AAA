import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from fuzzy.rules import diet_simulator

# Step 1: Set input values
diet_simulator.input['bmi'] = 23.0          # Normal
diet_simulator.input['age'] = 28            # Young/Adult boundary
diet_simulator.input['activity'] = 7        # High
diet_simulator.input['satiety'] = 4         # Medium-High satiety

# Step 2: Compute fuzzy output
diet_simulator.compute()

# Step 3: Output result
recommendation_value = diet_simulator.output['diet_type']
print(f"Recommended diet score: {recommendation_value:.3f}")

# Interpretation helper (optional)
if recommendation_value <= 0.33:
    print("Suggested Diet Type: Vegan")
elif recommendation_value <= 0.6:
    print("Suggested Diet Type: Balanced")
elif recommendation_value <= 0.85:
    print("Suggested Diet Type: High Protein")
else:
    print("Suggested Diet Type: Low Carb")
