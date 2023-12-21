import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import warnings

# import FutureWarnings from sns.histplot
warnings.simplefilter(action='ignore', category=FutureWarning)

# make ../plots/ directory if it doesn't exist
if not os.path.exists('plots/'):
    os.makedirs('plots/')

# load dataset A
data = pd.read_csv('data/A_NoiseAdded.csv')

# extracting the first 20 features for plotting
features = data.columns[1:21]

# creating figure with axes objects
fig, axes = plt.subplots(5, 4, figsize=(20, 15))
axes = axes.flatten()

# plotting histogram with kernel density for each feature
for i, feature in enumerate(features):
    sns.histplot(data[feature], bins=20, kde=True, ax=axes[i], stat="density")
    sns.kdeplot(data[feature], color='crimson', ax=axes[i], clip=[0,None])
    axes[i].set_title(feature)
    axes[i].set_xlabel('Value')
    axes[i].set_ylabel('Density')
    axes[i].set_xticks([0,1,2,3,4,5,6,7])

plt.tight_layout()

filepath = 'plots/1a_distributions.png'
fig.savefig(filepath)
print(f'First 20 feature distributions saved in {filepath}')
