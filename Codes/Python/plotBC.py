import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Read the data from the text file
data = pd.read_csv('../Sbatch/BreakthroughCurve.txt', sep='\t', header=None, names=['bc_time', 'cum_part'])

# Step 2: Plot the data
plt.plot(data['bc_time'], data['cum_part'])
plt.xlabel('Time step')
plt.ylabel('CDF')
plt.title('Breakthorugh curve')
plt.grid()
plt.show()