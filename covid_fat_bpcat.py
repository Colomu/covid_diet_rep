import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data
raw_data = pd.read_csv('C:/Users/olomuc/OneDrive - MHRA/COVID_ML_MODELS/covid_fat_death2.csv')

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 6), gridspec_kw={"width_ratios": [2, 3]})

# Create a boxplot on the first subplot
sns.boxplot(x='Continent', y='Deaths', data=raw_data, ax=ax1)
ax1.set_title('Deaths by Continent - Boxplot')

# Create a swarmplot on the second subplot
sns.swarmplot(x='Continent', y='Deaths', data=raw_data, ax=ax2)
ax2.set_title('Deaths by Continent - Swarmplot')

# wrap the text of the x-axis font
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha="right", fontsize=8)
ax1.tick_params(axis="x", pad=5)

ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha="right", fontsize=8)
ax2.tick_params(axis="x", pad=5)

# Save the plot to a PNG file with larger size
plt.savefig('box_swarm_plot.png', dpi=300)

# Print a message indicating that the plot was saved
print('Boxplot and swarmplot saved to box_swarm_plot.png')
