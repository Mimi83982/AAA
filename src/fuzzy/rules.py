# src/fuzzy/rules.py
# ------------------------------------------------------------------
# Fuzzy engine for SmartDietNF – maps BMI, age, activity, satiety to
# four diet‑type consequents: vegan, balanced, high_protein, low_carb
# ------------------------------------------------------------------
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# ────────────────────────── UNIVERSES ─────────────────────────────
bmi_range      = np.arange(10, 41, 0.1)   # 10‑40
age_range      = np.arange(15, 81, 1)
activity_range = np.arange(0, 11, 1)      # 0‑10 (mapped)
satiety_range  = np.arange(0, 6, 1)       # 0‑5 Likert
diet_range     = np.arange(0, 1.1, 0.1)   # fuzzy output 0‑1

# ───────────────────────── VARIABLES ──────────────────────────────
bmi      = ctrl.Antecedent(bmi_range, 'bmi')
age      = ctrl.Antecedent(age_range, 'age')
activity = ctrl.Antecedent(activity_range, 'activity')
satiety  = ctrl.Antecedent(satiety_range, 'satiety')
diet     = ctrl.Consequent(diet_range,  'diet_type')

# ─────────────────── MEMBERSHIP FUNCTIONS ─────────────────────────
bmi['underweight'] = fuzz.trimf(bmi_range, [10, 10, 18.5])
bmi['normal']      = fuzz.trimf(bmi_range, [18, 22, 25])
bmi['overweight']  = fuzz.trimf(bmi_range, [24, 30, 35])
bmi['obese']       = fuzz.trimf(bmi_range, [30, 35, 40])

age['young']   = fuzz.trimf(age_range, [15, 20, 30])
age['adult']   = fuzz.trimf(age_range, [25, 40, 55])
age['elderly'] = fuzz.trimf(age_range, [50, 65, 80])

activity['low']    = fuzz.trimf(activity_range, [0, 2, 4])
activity['medium'] = fuzz.trimf(activity_range, [3, 5, 7])
activity['high']   = fuzz.trimf(activity_range, [6, 8, 10])

satiety['low']    = fuzz.trimf(satiety_range, [0, 1, 2])
satiety['medium'] = fuzz.trimf(satiety_range, [1, 3, 4])
satiety['high']   = fuzz.trimf(satiety_range, [3, 5, 5])

diet['vegan']        = fuzz.trimf(diet_range, [0.0, 0.0, 0.33])
diet['balanced']     = fuzz.trimf(diet_range, [0.2, 0.4, 0.6])
diet['high_protein'] = fuzz.trimf(diet_range, [0.5, 0.7, 0.9])
diet['low_carb']     = fuzz.trimf(diet_range, [0.8, 1.0, 1.0])

# ───────────────────── RULE‑SET ───────────────────────────────────
rules = [
    # UNDERWEIGHT
    ctrl.Rule(bmi['underweight'] & activity['low'],    diet['balanced']),
    ctrl.Rule(bmi['underweight'] & activity['medium'], diet['high_protein']),
    ctrl.Rule(bmi['underweight'] & activity['high'],   diet['high_protein']),
    ctrl.Rule(bmi['underweight'] & satiety['low'],     diet['high_protein']),

    # NORMAL BMI
    ctrl.Rule(bmi['normal'] & activity['low']    & satiety['high'],   diet['balanced']),
    ctrl.Rule(bmi['normal'] & activity['medium'] & satiety['medium'], diet['balanced']),
    ctrl.Rule(bmi['normal'] & activity['high'],                    diet['high_protein']),
    ctrl.Rule(bmi['normal'] & age['young']   & satiety['medium'],  diet['vegan']),
    ctrl.Rule(bmi['normal'] & age['elderly'] & activity['low'],    diet['balanced']),

    # OVERWEIGHT
    ctrl.Rule(bmi['overweight'] & activity['low'],    diet['low_carb']),
    ctrl.Rule(bmi['overweight'] & activity['medium'], diet['balanced']),
    ctrl.Rule(bmi['overweight'] & activity['high'],   diet['balanced']),
    ctrl.Rule(bmi['overweight'] & satiety['low'],     diet['balanced']),
    ctrl.Rule(bmi['overweight'] & age['elderly'],     diet['low_carb']),

    # OBESE
    ctrl.Rule(bmi['obese'] & activity['low'],    diet['low_carb']),
    ctrl.Rule(bmi['obese'] & activity['medium'], diet['low_carb']),
    ctrl.Rule(bmi['obese'] & activity['high'],   diet['high_protein']),
    ctrl.Rule(bmi['obese'] & satiety['high'],    diet['low_carb']),
]

diet_ctrl      = ctrl.ControlSystem(rules)
diet_simulator = ctrl.ControlSystemSimulation(diet_ctrl)

# ──────────────────── HELPER FUNCTIONS ────────────────────────────
def _map_activity_level(level: str) -> int:
    level_map = {'Low': 2, 'Medium': 5, 'High': 8}
    return level_map.get(level, 5)

def _clip(val, lo, hi):
    """Force value into [lo, hi] to avoid out‑of‑universe errors."""
    return max(lo, min(val, hi))

def get_fuzzy_memberships(profile: dict, bmi_val: float):
    """Inject crisp values into the simulator — safely clamped."""
    diet_simulator.reset()
    diet_simulator.input['bmi']      = _clip(bmi_val, 10, 40)
    diet_simulator.input['age']      = _clip(profile['age'], 15, 80)
    diet_simulator.input['activity'] = _clip(_map_activity_level(profile['activity_level']), 0, 10)
    diet_simulator.input['satiety']  = _clip(profile.get('satiety', 3), 0, 5)
    return diet_simulator.input

def get_fuzzy_output() -> dict:
    """
    Compute fuzzy output safely; if anything goes wrong,
    return equal weights to avoid app crash.
    """
    try:
        diet_simulator.compute()
        crisp = diet_simulator.output['diet_type']
    except Exception as e:
        print("⚠️ Fuzzy compute failed:", e)
        return {'diet_type': {k: 0.25 for k in ['vegan','balanced','high_protein','low_carb']}}

    out = {
        'vegan':        fuzz.interp_membership(diet.universe, diet['vegan'].mf, crisp),
        'balanced':     fuzz.interp_membership(diet.universe, diet['balanced'].mf, crisp),
        'high_protein': fuzz.interp_membership(diet.universe, diet['high_protein'].mf, crisp),
        'low_carb':     fuzz.interp_membership(diet.universe, diet['low_carb'].mf, crisp),
    }
    return {'diet_type': out}
