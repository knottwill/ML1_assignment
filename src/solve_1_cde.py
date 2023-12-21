import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import warnings
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.decomposition import PCA
import pandas as pd
from collections import Counter

# import FutureWarnings from sns.histplot
warnings.simplefilter(action='ignore', category=FutureWarning)

# make plots/ directory if it doesn't exist
if not os.path.exists('plots/'):
    os.makedirs('plots/')

# load dataset A
data = pd.read_csv('data/A_NoiseAdded.csv')
X = data.drop(columns=['Unnamed: 0', 'classification'])

# splitting the data into two equal training sets randomly
train_set_1, train_set_2 = train_test_split(X, test_size=0.5, random_state=42)

# applying KMeans clustering with default parameters to each training set
kmeans_1 = KMeans(random_state=42).fit(train_set_1)
kmeans_2 = KMeans(random_state=42).fit(train_set_2)

# predicting clusters for the unused data in each case
clusters_1_on_2 = kmeans_1.predict(train_set_2)
clusters_2_on_1 = kmeans_2.predict(train_set_1)

# combining the predicted clusters
combined_clusters_1 = np.concatenate([kmeans_1.labels_, clusters_1_on_2])
combined_clusters_2 = np.concatenate([clusters_2_on_1, kmeans_2.labels_])

# getting cluster counts for both models
cluster_counts_1 = Counter(combined_clusters_1)
cluster_counts_2 = Counter(combined_clusters_2)
print("Cluster Sizes for kmeans_1:", cluster_counts_1)
print("Cluster Sizes for kmeans_2:", cluster_counts_2)

# creating and plotting the contingency table
contingency_table = confusion_matrix(combined_clusters_1, combined_clusters_2)

ConfusionMatrixDisplay(contingency_table).plot()
plt.xlabel('Model 2 cluster labels')
plt.ylabel('Model 1 cluster labels')
filepath = 'plots/1c_contingency.png'
plt.savefig(filepath)
print(f"Contingency table saved in {filepath}")

# performing K-Means clustering with n_clusters=2 (BEFORE PCA)
X_labels = KMeans(n_clusters=2, random_state=42).fit_predict(X)

# applying PCA to reduce the data to 2 dimensions
pca_clusters = PCA(n_components=2)
X_pca = pca_clusters.fit_transform(X)

# performing K-Means clustering AFTER PCA
X_pca_labels = KMeans(n_clusters=2, random_state=42).fit_predict(X_pca)

# showing that KMeans before and after PCA labels is the same
assert np.all(X_labels == X_pca_labels)

# creating a DataFrame for the PCA components and cluster labels
pca_clusters_df = pd.DataFrame(data=X_pca, columns=['Principal Component 1', 'Principal Component 2'])
pca_clusters_df['K-Means Clusters'] = X_labels

# plotting the PCA with clusters identified
fig, ax = plt.subplots(figsize=(10, 8))
sns.scatterplot(
    x='Principal Component 1', 
    y='Principal Component 2', 
    hue='K-Means Clusters', 
    data=pca_clusters_df,
    palette='viridis',
    ax=ax
)
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.grid(True)

filepath = 'plots/1e_PCA.png'
plt.savefig(filepath)
print(f"PCA plot saved in {filepath}")

