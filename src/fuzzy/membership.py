import numpy as np
import skfuzzy as fuzz

def generate_bmi_fuzzy_sets():
    # Define the universe of discourse for BMI
    bmi_range = np.arange(10, 41, 0.1)

    # Define fuzzy membership functions for each category
    underweight = fuzz.trimf(bmi_range, [10, 10, 18.5])
    normal = fuzz.trimf(bmi_range, [18, 22, 25])
    overweight = fuzz.trimf(bmi_range, [24, 27, 30])
    obese = fuzz.trimf(bmi_range, [29, 35, 40])

    return {
        'x': bmi_range,
        'underweight': underweight,
        'normal': normal,
        'overweight': overweight,
        'obese': obese
    }

def generate_activity_fuzzy_sets():
    # Activity level from 0 to 10 (arbitrary scale)
    x_activity = np.arange(0, 11, 0.1)

    # Membership functions
    low = fuzz.trimf(x_activity, [0, 0, 3])
    medium = fuzz.trimf(x_activity, [2, 5, 7])
    high = fuzz.trimf(x_activity, [6, 10, 10])

    return {
        'x': x_activity,
        'low': low,
        'medium': medium,
        'high': high
    }

def generate_age_fuzzy_sets():
    # Age range from 10 to 80 years
    x_age = np.arange(10, 81, 1)

    # Membership functions
    young = fuzz.trimf(x_age, [10, 18, 30])
    adult = fuzz.trimf(x_age, [25, 40, 60])
    elderly = fuzz.trimf(x_age, [55, 70, 80])

    return {
        'x': x_age,
        'young': young,
        'adult': adult,
        'elderly': elderly
    }

def generate_satiety_fuzzy_sets():
    # Satiety scale: 1 (Hungry) to 5 (Overfull)
    x_satiety = np.arange(1, 5.1, 0.1)

    # Define fuzzy membership functions
    low = fuzz.trimf(x_satiety, [1, 1, 2.5])
    medium = fuzz.trimf(x_satiety, [2, 3, 4])
    high = fuzz.trimf(x_satiety, [3.5, 5, 5])

    return {
        'x': x_satiety,
        'low': low,
        'medium': medium,
        'high': high
    }