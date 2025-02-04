debug = False
if not debug:
    from IPython import get_ipython
    get_ipython().run_line_magic('reset', '-f')
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from sklearn.linear_model import LinearRegression

save = False

loadFinalPositions = np.load('partialAdsorption.npz')
for name, value in (loadFinalPositions.items()):
    globals()[name] = value

# Normalised distribution of non-absorbed particles in time
diffusionLimitedSurvTimeDistNorm = plt.figure(figsize=(8, 8))
plt.plot(timeStep, liveParticlesInTimeNormAds, 'b*')
plt.title("Non-absorbed particle normalised in time")
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Time')
plt.ylabel('Non-absorbed particles normalised')
plt.grid(True, which="major", linestyle='-', linewidth=0.7, color='black')
plt.grid(True, which="minor", linestyle=':', linewidth=0.5, color='gray')
plt.tight_layout()