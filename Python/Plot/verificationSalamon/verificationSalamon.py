debug = False
if not debug:
    from IPython import get_ipython
    get_ipython().run_line_magic('reset', '-f')
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from sklearn.linear_model import LinearRegression

save = False

#    loadFinalPositions = np.load('Dl01Dr01Rl0Rr0.npz')
#    for name, value in (loadFinalPositions.items()):
#        globals()[name] = value

#    loadFinalPositions = np.load('Dl01Dr001Rl0Rr0.npz')
#    for name, value in (loadFinalPositions.items()):
#        globals()[name] = value

#    loadFinalPositions = np.load('Dl01Dr01RlPlRrPr.npz')
#    for name, value in (loadFinalPositions.items()):
#        globals()[name] = value

loadFinalPositions = np.load('Dl01Dr001RlPlRrPr.npz')
for name, value in (loadFinalPositions.items()):
    globals()[name] = value

#    loadFinalPositions = np.load('Dl01Dr001RlPlRrPr1e5ts.npz')
#    for name, value in (loadFinalPositions.items()):
#        globals()[name] = value

# loadMatrixDiffVer = np.load('matrixDiffusionVerification.npz')
# for name, value in (loadMatrixDiffVer.items()):
#     globals()[name] = value

# loadTestSalamon = np.load('testSalamon.npz')
# for name, value in (loadTestSalamon.items()):
#     globals()[name] = value

# Final particles's positions
finalPositions = plt.figure(figsize=(8, 8))
if matrixDiffVerification:
    plt.plot(x, y, 'b*')
    plt.plot([lbx, rbx, rbx, lbx, lbx], [lby, lby, uby, uby, lby], color='black', linewidth=2)
    plt.scatter(x0, y0, s=2, c='purple', alpha=1, edgecolor='none', marker='o')
    if (reflectedLeft).any() & (reflectedRight).any():
        plt.plot([cbx, cbx], [lby, uby], color='orange', linewidth=3, linestyle='--')
    histoMatriDiff = plt.figure(figsize=(8, 8))
    hDist, hBins = np.histogram(x, np.linspace(lbx, rbx, 100), density=True)
    plt.bar(hBins[:-1], hDist, width=np.diff(hBins), edgecolor="black", align="edge")
    plt.axvline(x=cbx, color='orange', linestyle='-', linewidth=2)
else:
    # plt.plot(xPath[:, -1], yPath[:, -1], 'b*')
    plt.plot(x, y, 'b*')
    plt.axvline(x=xInit, color='yellow', linestyle='--', linewidth=2)
    plt.axhline(y=uby, color='r', linestyle='--', linewidth=1)
    plt.axhline(y=lby, color='r', linestyle='--', linewidth=1)
    # for val in vInterval:
    #     plt.axvline(x=val, color='black', linestyle='--', linewidth=2)
    # for val in hInterval:
    #     plt.axhline(y=val, color='black', linestyle='--', linewidth=2)
plt.tight_layout()

# if save:
    # finalPositions.savefig("/home/eugenio/Github/IDAEA/Overleaf/WeeklyMeetingNotes/images/positionsDl01Dr01Rl0Rr0.png", format="png", bbox_inches="tight")
    # histoMatriDiff.savefig("/home/eugenio/Github/IDAEA/Overleaf/WeeklyMeetingNotes/images/histDl01Dr01Rl0Rr0.png", format="png", bbox_inches="tight")
    # finalPositions.savefig("/home/eugenio/Github/IDAEA/Overleaf/WeeklyMeetingNotes/images/positionsDl01Dr001Rl0Rr0.png", format="png", bbox_inches="tight")
    # histoMatriDiff.savefig("/home/eugenio/Github/IDAEA/Overleaf/WeeklyMeetingNotes/images/histDl01Dr001Rl0Rr0.png", format="png", bbox_inches="tight")
    # finalPositions.savefig("/home/eugenio/Github/IDAEA/Overleaf/WeeklyMeetingNotes/images/positionsDl01Dr01RlPlRrPr.png", format="png", bbox_inches="tight")
    # histoMatriDiff.savefig("/home/eugenio/Github/IDAEA/Overleaf/WeeklyMeetingNotes/images/histDl01Dr01RlPlRrPr.png", format="png", bbox_inches="tight")
    # finalPositions.savefig("/home/eugenio/Github/IDAEA/Overleaf/WeeklyMeetingNotes/images/positionsDl01Dr001RlPlRrPr.png", format="png", bbox_inches="tight")
    # histoMatriDiff.savefig("/home/eugenio/Github/IDAEA/Overleaf/WeeklyMeetingNotes/images/histDl01Dr001RlPlRrPr.png", format="png", bbox_inches="tight")
    # finalPositions.savefig("/home/eugenio/Github/IDAEA/Overleaf/WeeklyMeetingNotes/images/positionsDl01Dr001RlPlRrPr1e5ts.png", format="png", bbox_inches="tight")
    # histoMatriDiff.savefig("/home/eugenio/Github/IDAEA/Overleaf/WeeklyMeetingNotes/images/histDl01Dr001RlPlRrPr1e5ts.png", format="png", bbox_inches="tight")