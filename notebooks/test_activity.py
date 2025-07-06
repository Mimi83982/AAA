import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import matplotlib.pyplot as plt
from fuzzy.membership import generate_activity_fuzzy_sets

activity_sets = generate_activity_fuzzy_sets()
x = activity_sets['x']

plt.figure(figsize=(10, 5))
plt.plot(x, activity_sets['low'], label='Low')
plt.plot(x, activity_sets['medium'], label='Medium')
plt.plot(x, activity_sets['high'], label='High')

plt.title('Activity Level Membership Functions')
plt.xlabel('Activity Level')
plt.ylabel('Membership')
plt.legend()
plt.grid(True)
plt.show()
