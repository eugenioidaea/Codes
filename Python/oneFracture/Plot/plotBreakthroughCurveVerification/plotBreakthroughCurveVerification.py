debug = False
if not debug:
    from IPython import get_ipython
    get_ipython().run_line_magic('reset', '-f')
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from sklearn.linear_model import LinearRegression

save = False

loadSemiInfiniteDomain = np.load('semiInfiniteDomain1e3.npz')
for name, value in (loadSemiInfiniteDomain.items()):
    globals()[name] = value

pdfOfBtc = plt.figure(figsize=(8, 8))
plt.rcParams.update({'font.size': 20})
plt.plot(timeBinsLog, countsSemiInfLog, 'b*')
plt.plot(timeBinsLog, analPdfSemiInf, 'r-')
plt.xscale('log')
plt.yscale('log')
plt.ylim(1e-5, 1e-2)
plt.title('PDF of breakthrough curve')
plt.xlabel('Time step')
plt.ylabel('Normalised number of particles')
plt.grid(True, which="major", linestyle='-', linewidth=0.7, color='black')
plt.grid(True, which="minor", linestyle=':', linewidth=0.5, color='gray')
plt.tight_layout()

if save:
    pdfOfBtc.savefig("/home/eugenio/Github/IDAEA/Overleaf/WeeklyMeetingNotes/images/verificationSemi-infinite1e3.png", format="png", bbox_inches="tight")