import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))


import matplotlib.pyplot as plt
from fuzzy.membership import generate_bmi_fuzzy_sets

bmi_sets = generate_bmi_fuzzy_sets()
x = bmi_sets['x']
plt.figure(figsize=(10, 5))
plt.plot(x, bmi_sets['underweight'], label='Underweight')
plt.plot(x, bmi_sets['normal'], label='Normal')
plt.plot(x, bmi_sets['overweight'], label='Overweight')
plt.plot(x, bmi_sets['obese'], label='Obese')

plt.title('BMI Membership Functions')
plt.xlabel('BMI')
plt.ylabel('Membership')
plt.legend()
plt.grid(True)
plt.show()
