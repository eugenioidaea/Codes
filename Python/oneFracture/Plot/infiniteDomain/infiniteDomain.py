debug = False
if not debug:
    from IPython import get_ipython
    get_ipython().run_line_magic('reset', '-f')
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from sklearn.linear_model import LinearRegression

save = False

loadInfiniteDomain = np.load('infiniteDomain1e6.npz')
for name, value in (loadInfiniteDomain.items()):
    globals()[name] = value

# Binning for plotting the pdf from a Lagrangian vector
countsLog, binEdgesLog = np.histogram(particleRT, timeLogSpaced, density=True)
binCentersLog = (binEdgesLog[:-1] + binEdgesLog[1:]) / 2
plt.figure(figsize=(8, 8))
plt.plot(binCentersLog[countsLog!=0], countsLog[countsLog!=0], 'r*')
plt.xscale('log')
plt.yscale('log')
plt.title("Lagrangian PDF of the BTC")

# Spatial concentration profile at 'recordSpatialConc' time
spatialConcentration = plt.figure(figsize=(8, 8))
plt.rcParams.update({'font.size': 20})
plt.plot(binCenterSpace, countsSpace, 'b*')
plt.plot(binCenterSpace, yAnalytical, color='red', linestyle='-')
plt.axvline(x=x0, color='yellow', linestyle='--', linewidth=2)
if lbxOn:
    plt.axvline(x=lbx, color='b', linestyle='--', linewidth=2)
    plt.axvline(x=-lbx, color='b', linestyle='--', linewidth=2)
plt.xlabel('X position')
plt.ylabel('Normalised number of particles')
plt.title("Spatial concentration at t=100")
plt.tight_layout()

if save:
    spatialConcentration.savefig("/home/eugenio/Github/IDAEA/Overleaf/WeeklyMeetingNotes/images/verificationInfinite1e6.png", format="png", bbox_inches="tight")
    spatialConcentration.savefig("/home/eugenio/Github/IDAEA/Overleaf/WeeklyMeetingNotes/images/verificationInfinite1e5.png", format="png", bbox_inches="tight")