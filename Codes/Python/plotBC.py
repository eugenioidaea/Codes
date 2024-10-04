import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Read the data from the text file
data = pd.read_csv('../Sbatch/BreakthroughCurve.txt', sep='\t', header=None, names=['X', 'Y'])

# Step 2: Plot the data
plt.figure(figsize=(10, 6))
plt.plot(data['X'], data['Y'], marker='o', linestyle='-', color='b', label='Data Line')

# Step 3: Add labels and title
plt.xlabel('X Values')
plt.ylabel('Y Values')
plt.title('Plot of Two Columns from a Tab-Separated File')
plt.legend()
plt.grid()

# Step 4: Show the plot
plt.show()