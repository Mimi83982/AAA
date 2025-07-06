import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import matplotlib.pyplot as plt
from fuzzy.membership import generate_age_fuzzy_sets

age_sets = generate_age_fuzzy_sets()
x = age_sets['x']

plt.figure(figsize=(10, 5))
plt.plot(x, age_sets['young'], label='Young')
plt.plot(x, age_sets['adult'], label='Adult')
plt.plot(x, age_sets['elderly'], label='Elderly')

plt.title('Age Membership Functions')
plt.xlabel('Age')
plt.ylabel('Membership')
plt.legend()
plt.grid(True)
plt.show()
