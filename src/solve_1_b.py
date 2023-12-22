import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os
import warnings

# ignore FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# make plots/ directory if it doesn't exist
if not os.path.exists('plots/'):
    os.makedirs('plots/')

# load dataset A
data = pd.read_csv('data/A_NoiseAdded.csv')
X = data.drop(columns=['Unnamed: 0', 'classification'])

# apply PCA to reduce the data to 2 dimensions
pca = PCA(n_components=2)
principal_components = pca.fit_transform(X)

# convert to a DataFrame for easier plotting
pca_df = pd.DataFrame(data = principal_components, columns = ['Principal Component 1', 'Principal Component 2'])

# plot the 2D PCA visualisation
fig, ax = plt.subplots(figsize=(10, 8))
sns.scatterplot(x='Principal Component 1', y='Principal Component 2', data=pca_df, marker='x', color='green', ax=ax)
ax.grid(True)
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')

filepath = 'plots/1b_PCA.png'
fig.savefig(filepath)
print(f'PCA visualisation saved in {filepath}')
