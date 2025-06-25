import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from scipy.optimize import minimize
from scipy.optimize import curve_fit

# Functions #################################################
def load_mas_file(filename):
    try:
        with open(filename,"r") as fp:
            line = fp.readline()
        
        keynames = line.split(",")
        keynames = [key.strip('\n') for key in keynames]
        keynames = [key.strip('"') for key in keynames]
        keynames = [key.strip(' "') for key in keynames]
        dictionary = dict.fromkeys(keynames, None)
        data = np.genfromtxt(filename,skip_header = 1)

        for i,key in enumerate(keynames):
            dictionary[key] = data[:,i]
        df = pd.DataFrame.from_dict(dictionary)
        return df
    except:
        pass

# def CJsolFor(t_array, n_terms, x):
#     nArr = np.arange(1, n_terms + 1)
#     t = t_array
#     J = np.zeros(len(time))
#     for ti in range(0, len(t)):
#         cos_part = np.cos(nArr * np.pi * x / L)
#         exp_part = np.exp(-Deff * (nArr**2) * np.pi**2 * t[ti] / L**2)
#         J[ti] = Deff*C1/L * (1 + 2*np.sum(cos_part*exp_part))
#     J = J/np.max(J)    
#     return J

def CJsolVector(t_array, n_terms, x, D):
    nArr = np.arange(1, n_terms + 1).reshape(-1, 1)  # shape (n_terms, 1)
    t = t_array.reshape(1, -1)  # shape (1, num_times)
    
    cos_part = np.cos(nArr * np.pi * x / L)  # shape (n_terms, 1)
    exp_part = np.exp(-D * (nArr**2) * np.pi**2 * t / L**2)  # shape (n_terms, num_times)
    
    sum_terms = cos_part * exp_part  # shape (n_terms, num_times)
    sum_result = np.sum(sum_terms, axis=0)  # sum over n
    
    J = D*C1/L * (1 + 2*sum_result) * Atot # shape (num_times,)
    J = J/np.max(J)
    return J

# Least squares
def lsqJvec(Deff):
    nArr = np.arange(1, n + 1).reshape(-1, 1)  # shape (n_terms, 1)
    t = time.reshape(1, -1)  # shape (1, num_times)
    
    cos_part = np.cos(nArr * np.pi * x / L)  # shape (n_terms, 1)
    exp_part = np.exp(-Deff * (nArr**2) * np.pi**2 * t / L**2)  # shape (n_terms, num_times)
    
    sum_terms = cos_part * exp_part  # shape (n_terms, num_times)
    sum_result = np.sum(sum_terms, axis=0)  # sum over n
    
    Jopt = Atot*Deff*C1/L * (1 + 2*sum_result)  # shape (num_times,)
    Jopt = Jopt/np.max(Jopt)
    return Jopt
def objective(Deff):
    Jobj = lsqJvec(Deff[0]) # Deff is an array from optimizer
    return np.sum((Jsim - Jobj) ** 2)

# Curve fitting
def fitJvec(time, Deff):
    nArr = np.arange(1, n + 1).reshape(-1, 1)   # shape (n_terms, 1)
    t = time.reshape(1, -1)                     # shape (1, num_times)

    cos_part = np.cos(nArr * np.pi * x / L)     # shape (n_terms, 1)
    exp_part = np.exp(-Deff * (nArr**2) * np.pi**2 * t / L**2)  # (n_terms, num_times)

    sum_terms = cos_part * exp_part
    sum_result = np.sum(sum_terms, axis=0)

    Jfit = Atot * Deff * C1 / L * (1 + 2 * sum_result)
    Jfit = Jfit/np.max(Jfit)
    return Jfit

# Parsing intersection file ############################################
import pandas as pd
file_path = "output/dfnGen_output/intersection_list.dat"

column_names = ['f1', 'f2', 'x', 'y', 'z', 'length']

dfInter = pd.read_csv(file_path, delim_whitespace=True, skiprows=1, names=column_names)
filtered_df = dfInter[dfInter['f2'].isin([-1, -2])]

name_f2_minus1 = filtered_df[filtered_df['f2'] == -1]['f1']+6
length_f2_minus1 = filtered_df[filtered_df['f2'] == -1]['length'].sum()
length_f2_minus2 = filtered_df[filtered_df['f2'] == -2]['length'].sum()

print(f"Total length where f2 = top: {length_f2_minus1}")
print(f"Total length where f2 = bottom: {length_f2_minus2}")

# Parsing aperture file ################################################
keys = set(f"-{i}" for i in name_f2_minus1)
with open("output/aperture.dat", "r") as file:
    width = []
    for line in file:
        if line.strip() == "aperture" or not line.strip():
            continue  # skip header or blank lines
        parts = line.split()
        if parts[0] in keys:
            width.append(float(parts[-1]))  # last column
width = np.array(width)

# Load numerical solution ##############################################
mas_filename = "output/dfn_diffusion_no_flow-mas.dat"
df = load_mas_file(mas_filename)

# Input paramenters ####################################################
Dmol = 4.5e-7 # [m2/s] DIFFUSION_COEFFICIENT from PFLOTRAN input file
time = np.array(df['Time [y]'])*86400*365 # [s]
Jsim = -np.array(df['OUTFLOW TRACER [mol/y]'])/(86400*365) # [mol/s]
C1 = 1e3 # [mol/m3]
Atot = width.sum()*length_f2_minus1 # [m2]
L = 20 # [m]
n = 100 # [-]
x = L # Distance between inlet and btc record section # [m]
Deff = Jsim[-1]*L/(C1*Atot) # 4.5e-6 # [m2/s]

Jsim = Jsim/np.max(Jsim)

# Compute analytical solution ##########################################
# Jfor = CJsolFor(time, n, x)
Jvec = CJsolVector(time, n, x, Dmol)

# OPTIMISTAION ###############################################################################
# Initial guess for Deff
initial_guess = [3.835337584289503e-08]
# Run optimization
result = minimize(objective, initial_guess, method='Nelder-Mead')
# Optimal Deff
lsqDeff = result.x[0]
# Initial guess for Deff
initial_guess = [Deff]

# Fit the model
popt, pcov = curve_fit(fitJvec, time, Jsim, p0=initial_guess)
# Extract best-fit Deff
curveFitDeff = popt[0]

# PLOTS ################################################################
# Analytical solution
# fig, ax = plt.subplots(figsize = (8,6))
# plt.rcParams.update({'font.size': 20})
# ax.plot(time, Jvec*Atot, label='Analytical')
# plt.xlabel('Time [s]')
# plt.ylabel('Outflowing tracer [mol/s]')
# plt.xscale('log')
# # plt.yscale('log')
# plt.legend(loc='best')
# plt.grid(True)
# plt.show()

# Numerical vs analytical
fig, ax = plt.subplots(figsize = (8,6))
plt.rcParams.update({'font.size': 20})
ax.plot(time, Jsim, 'o', markerfacecolor='none', markeredgecolor='red', markersize='8', label='Numerical')
ax.plot(time, Jvec, color='orange', linewidth=3, label='Dmol')
# ax.plot(time, Jfor*Atot/(86400*365), color='blue', linewidth=3, label='Analytical')
# ax.plot(time, Jfor*Atot, color='blue', linewidth=3, label='Analytical')
# plt.title('Breakthrough curve')
plt.xlabel('Time [s]')
plt.ylabel('Outflowing tracer [-]')
plt.xscale('log')
# plt.yscale('log')
plt.legend(loc='best')
plt.grid(True)
plt.show()
diff = np.linalg.norm(Jsim - Jvec, ord=np.inf)
print(f"diff = {diff:.6e}")

# Numerical vs analytical vs optimised ######################################################
fig, ax = plt.subplots(figsize = (8,6))
plt.rcParams.update({'font.size': 20})
ax.plot(time, Jsim, 'o', markerfacecolor='none', markeredgecolor='red', markersize='8', label='Numerical')
ax.plot(time, Jvec, color='orange', linewidth=3, label='Dmol')
ax.plot(time, lsqJvec(lsqDeff), '-*', color='green', markersize='8', label='lsq')
# ax.plot(time, fitJvec(time, curveFitDeff), '*', markerfacecolor='none', markeredgecolor='pink', markersize='5', label='fit')
plt.xlabel('Time [s]')
plt.ylabel('Outflowing tracer [-]')
plt.xscale('log')
# plt.yscale('log')
plt.legend(loc='best')
plt.grid(True)
plt.show()

print(f"Dmol [m2/s] = {Dmol:.4e}")
print(f"Deff [m2/s] = Jsim[-1]*L/(C1*Atot) = {Deff:.4e}")
print(f"Initial guess : D0 [m2/s] = {initial_guess[0]:.4e}")
print(f"Deff_LSQ [m2/s] = {lsqDeff:.4e}")
# print(f"Curve fit Deff [m2/s]: {curveFitDeff:.4e}")
print(f"Tortuosity: Dmol/Deff_LSQ = {Dmol/lsqDeff:.4}")

# # cNorm = -1*df['OUTFLOW TRACER [mol/y]']/np.sum(-1*df['OUTFLOW TRACER [mol/y]'])
# cPlateau1 = np.array(-1*df['OUTFLOW TRACER [mol/y]']/np.max(-1*df['OUTFLOW TRACER [mol/y]']))
# time = np.array(df['Time [y]'])
# fig, ax = plt.subplots(figsize = (8,6))
# plt.rcParams.update({'font.size': 20})
# ax.plot(time, cPlateau1)
# plt.title('Plateau normalised BTC')
# plt.xlabel('Time [y]')
# plt.ylabel('Outflowing tracer [mol/y]')
# plt.xscale('log')
# plt.yscale('log')
# plt.xlim(min(time), max(time))
# plt.ylim(1e-5, 1.1)
# plt.grid(True)
# plt.show()
# 
# cc = (cPlateau1[:-1]+cPlateau1[1:])/2
# dt = np.diff(time)/np.max(time)
# tt = (time[:-1]+time[1:])/2
# m1c = np.sum(cc*dt)
# m2c = np.sum((cc)**2*dt)
# VarC = m2c-m1c**2
# m1t = np.sum(tt*cc*dt)/np.sum(cc*dt)
# m2t = np.sum(tt**2*cc*dt)/np.sum(cc*dt)
# VarT = m2t - m1t**2
# # mrtVar = np.sum((tt-m1t)**2*cc*dt)/np.sum(cc*dt)
# 
# Ddisp = m2t/(2*m1t)
# # Deff = Ddisp/advVel
# 
# fig, ax = plt.subplots(figsize = (8,6))
# plt.rcParams.update({'font.size': 20})
# ax.plot(df['Time [y]'], 1-cPlateau1)
# plt.title('Complementary plateau norm BTC')
# plt.xlabel('Time [y]')
# plt.ylabel('Outflowing tracer [mol/y]')
# plt.xscale('log')
# plt.yscale('log')
# plt.xlim(1, max(time))
# plt.ylim(1e-5, 1.1)
# plt.grid(True)
# plt.show()