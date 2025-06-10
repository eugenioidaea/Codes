import numpy as np
import pandas as pd
import matplotlib.pylab as plt

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

mas_filename = "output/dfn_diffusion_no_flow-mas.dat"

df = load_mas_file(mas_filename)

Jsim = np.array(df['OUTFLOW TRACER [mol/y]'])
C1 = 1
Atot = 2 # 3.46410e-05*10.0
L = 20
Deff = -Jsim[-1]*L/(C1*Atot)
n = 50
time = np.array(df['Time [y]'])
x = 20.0  # Distance between inlet and btc record section

def CJsolFor(t_array, n_terms, x):
    n = np.arange(1, n_terms + 1)
    t = t_array
    J = np.zeros(len(time))

    for ti in range(0, len(t)):
        cos_part = np.cos(n * np.pi * x / L)
        exp_part = np.exp(-Deff * (n**2) * np.pi**2 * t[ti] / L**2)
        J[ti] = Deff*C1/L + (2*C1/L) * np.sum(cos_part*exp_part)

    return J
Jfor = CJsolFor(time, n, x)

def CJsolVector(t_array, n_terms, x):
    n = np.arange(1, n_terms + 1).reshape(-1, 1)  # shape (n_terms, 1)
    t = t_array.reshape(1, -1)  # shape (1, num_times)
    
    cos_part = np.cos(n * np.pi * x / L)  # shape (n_terms, 1)
    exp_part = np.exp(-Deff * (n**2) * np.pi**2 * t / L**2)  # shape (n_terms, num_times)
    
    sum_terms = cos_part * exp_part  # shape (n_terms, num_times)
    sum_result = np.sum(sum_terms, axis=0)  # sum over n
    
    J = Deff*C1/L + (2*C1/L) * sum_result  # shape (num_times,)
    return J
Jvec = CJsolVector(time, n, x)

fig, ax = plt.subplots(figsize = (8,6))
plt.rcParams.update({'font.size': 20})
ax.plot(df['Time [y]'], -1*df['OUTFLOW TRACER [mol/y]'], 'o', markerfacecolor='none', markeredgecolor='red', markersize='5', label='Numerical')
# ax.plot(time, Jvec*Atot)
ax.plot(time, Jfor*Atot, color='blue', linewidth=3, label='Analytical')
# plt.title('Breakthrough curve')
plt.xlabel('Time [y]')
plt.ylabel('Outflowing tracer [mol/y]')
plt.xscale('log')
# plt.yscale('log')
plt.legend(loc='best')
plt.grid(True)
plt.show()

# cNorm = -1*df['OUTFLOW TRACER [mol/y]']/np.sum(-1*df['OUTFLOW TRACER [mol/y]'])
cPlateau1 = np.array(-1*df['OUTFLOW TRACER [mol/y]']/np.max(-1*df['OUTFLOW TRACER [mol/y]']))
time = np.array(df['Time [y]'])
fig, ax = plt.subplots(figsize = (8,6))
plt.rcParams.update({'font.size': 20})
ax.plot(time, cPlateau1)
plt.title('Plateau normalised BTC')
plt.xlabel('Time [y]')
plt.ylabel('Outflowing tracer [mol/y]')
plt.xscale('log')
plt.yscale('log')
plt.xlim(min(time), max(time))
plt.ylim(1e-5, 1.1)
plt.grid(True)
plt.show()

cc = (cPlateau1[:-1]+cPlateau1[1:])/2
dt = np.diff(time)/np.max(time)
tt = (time[:-1]+time[1:])/2
m1c = np.sum(cc*dt)
m2c = np.sum((cc)**2*dt)
VarC = m2c-m1c**2
m1t = np.sum(tt*cc*dt)/np.sum(cc*dt)
m2t = np.sum(tt**2*cc*dt)/np.sum(cc*dt)
VarT = m2t - m1t**2
# mrtVar = np.sum((tt-m1t)**2*cc*dt)/np.sum(cc*dt)

Ddisp = m2t/(2*m1t)
# Deff = Ddisp/advVel

fig, ax = plt.subplots(figsize = (8,6))
plt.rcParams.update({'font.size': 20})
ax.plot(df['Time [y]'], 1-cPlateau1)
plt.title('Complementary plateau norm BTC')
plt.xlabel('Time [y]')
plt.ylabel('Outflowing tracer [mol/y]')
plt.xscale('log')
plt.yscale('log')
plt.xlim(1, max(time))
plt.ylim(1e-5, 1.1)
plt.grid(True)
plt.show()