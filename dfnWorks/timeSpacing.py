import numpy as np

# Original array
original_seconds = np.array([
    3.15360000e+06, 9.46080000e+06, 2.20752000e+07, 4.73040000e+07,
    9.77616000e+07, 1.98676800e+08, 4.00507200e+08, 8.04168000e+08,
    1.61148960e+09, 3.22613280e+09, 6.37973280e+09, 9.53333280e+09,
    1.26869328e+10, 1.58405328e+10, 1.89941328e+10, 2.21477328e+10,
    2.53013328e+10, 2.84549328e+10, 3.15360000e+10
])

original = original_seconds/(86400*365)

# Define the densified interval range
dense_min = 1e7/(86400*365)
dense_max = 1e9/(86400*365)

# Extract parts outside the densified range
before = original[original < dense_min]
after = original[original > dense_max]

# Create a denser array within the interval
dense_part = np.logspace(np.log10(dense_min), np.log10(dense_max), num=40)

# Concatenate and sort
new_array = np.sort(np.concatenate([before, dense_part, after]))

# Optional: print or inspect
print(np.array2string(new_array, precision=2, separator=' '))