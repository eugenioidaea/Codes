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

fig,ax = plt.subplots(figsize = (8,6))
ax.plot(df['Time [y]'], -1*df['OUTFLOW TRACER [mol/y]'])
plt.title('Breakthrough curve')
plt.xlabel('Time [y]')
plt.ylabel('Outflowing tracer [mol/y]')
plt.xscale('log')
plt.grid(True)
plt.show()

# cNorm = -1*df['OUTFLOW TRACER [mol/y]']/np.sum(-1*df['OUTFLOW TRACER [mol/y]'])
cPlateau1 = np.array(-1*df['OUTFLOW TRACER [mol/y]']/np.max(-1*df['OUTFLOW TRACER [mol/y]']))
time = np.array(df['Time [y]'])
fig,ax = plt.subplots(figsize = (8,6))
ax.plot(time, cPlateau1)
plt.title('Plateau normalised BTC')
plt.xlabel('Time [y]')
plt.ylabel('Outflowing tracer [mol/y]')
plt.xscale('log')
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

fig,ax = plt.subplots(figsize = (8,6))
ax.plot(df['Time [y]'], 1-cPlateau1)
plt.title('Complementary plateau norm BTC')
plt.xlabel('Time [y]')
plt.ylabel('Outflowing tracer [mol/y]')
plt.xscale('log')
plt.grid(True)
plt.show()