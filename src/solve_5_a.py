import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import os
import warnings
warnings.filterwarnings('ignore')
import umap

from utils import random_forest_imputation

# import FutureWarnings from sns.histplot
warnings.simplefilter(action='ignore', category=FutureWarning)

# make plots/ and data/ directories if they don't exist
if not os.path.exists('plots/'):
    os.makedirs('plots/')
if not os.path.exists('data/'):
    os.makedirs('data/')

# data = pd.read_csv('data/processed_for_lasso.csv')
data = pd.read_csv('data/ADS_BaselineDataset.csv')
X = data.drop(['Unnamed: 0','type'], axis=1)
y = data['type']
seed = 42

#########################

###   Preprocessing   ###

#########################

# -----------------
# Dropping highly sparse features
# -----------------

# calculate sparsity of features
def calc_sparsity(feature):
    return (feature==0).sum() / len(feature)

sparsity = X.apply(calc_sparsity)

# remove highly sparse features
highly_sparse_features = sparsity[sparsity >= 0.95].index
X = X.drop(columns=highly_sparse_features)

# -----------------
# Correcting most extreme outlier values
# -----------------

# find most extreme outlier values
def find_outliers(feature):
    # calculate sparsity of feature
    sparsity = (feature==0).sum() / len(feature)

    # standardise feature (calculate z scores)
    z_scores = (feature - feature.mean())/feature.std()

    # if sparsity is <= 20%, threshold is 5
    if sparsity <= 0.2:
        return np.abs(z_scores) > 5
    
    # if a sparse feature, standardize just the non-zero instances of the 
    # feature and have a separate threshold
    nonzero = feature[feature != 0]
    nonzero_z_scores = (nonzero - nonzero.mean())/nonzero.std()
    return (np.abs(z_scores) > 10) | (np.abs(nonzero_z_scores) > 5)

outlier_vals = X.apply(find_outliers)

# set outlier values to NaN
X_outliers_removed = X[outlier_vals == False]

# impute the outlier values (which are now NaN)
print('Beginning imputation of outlier values...')
X_imputed = random_forest_imputation(X_outliers_removed, {'random_state': seed})
X_imputed = pd.DataFrame(X_imputed, columns = X_outliers_removed.columns)

print(f"{(outlier_vals).sum().sum()} outlier values imputed\n")

# ------------
# Normalisation
# ------------

norm = MinMaxScaler()
X_norm = norm.fit_transform(X_imputed)
X = pd.DataFrame(X_norm, columns=X_imputed.columns)

# save dataset
processed_data = pd.concat([data['Unnamed: 0'], X, y], axis=1)
filepath = 'data/5_preprocessed.csv'
processed_data.to_csv(filepath, index=False)
print(f"Pre-processed dataset saved in {filepath}\n")

##############################

###  Selecting n_clusters  ###
# (elbow and silhouette method)
##############################

# calculating the inertia (within-clusters sum of squares) for the Elbow method
inertia = []
K_range = range(1, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=seed).fit(X)
    inertia.append(kmeans.inertia_)

# calculating silhouette scores for Silhouette method
silhouette_scores = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=seed).fit(X)
    score = silhouette_score(X, kmeans.labels_)
    silhouette_scores.append(score)

# plotting both methods
fig, axes = plt.subplots(2, 1, figsize=(10, 12))

# elbow Plot
axes[0].plot(K_range, inertia, marker='o')
axes[0].set_xlabel('Number of clusters', fontsize=12)
axes[0].set_ylabel('Inertia', fontsize=12)
axes[0].grid(True)
axes[0].text(2.3, max(inertia) * 0.99, "(a)", fontsize=30, color='black') # text label

# silhouette scores Plot
axes[1].plot(range(2, 11), silhouette_scores, marker='o')
axes[1].set_xlabel('Number of clusters', fontsize=12)
axes[1].set_ylabel('Silhouette Score', fontsize=12)
axes[1].grid(True)
axes[1].text(3.2, max(silhouette_scores) * 0.85, "(b)", fontsize=30, color='black') # text label 

plt.tight_layout()

# save figure
filepath = 'plots/elbow_silhouette.png'
fig.savefig(filepath)
print(f'Elbow & Silhouette method plots saved in {filepath}\n')

########################################

###  KMeans and Spectral clustering  ###

########################################

X = processed_data.drop(['Unnamed: 0', 'type'], axis=1)

# silhouette scores plot found 2 to the optimal number of clusters
n_clusters = 2

# perform KMeans and Spectral clustering, each with two clusters
km = KMeans(n_clusters=n_clusters, random_state=seed)
kmeans_labels = km.fit_predict(X)

spectral = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', random_state=seed)
spectral_labels = spectral.fit_predict(X)

# contingency matrix
conf_matrix = confusion_matrix(kmeans_labels, spectral_labels)
ConfusionMatrixDisplay(conf_matrix).plot(cmap="Blues")
plt.ylabel("KMeans Cluster Labels")
plt.xlabel("Spectral Cluster Labels")

# save contingency table
filepath = 'plots/kmeans_spectral.png'
plt.savefig(filepath)
print(f'KMeans vs Spectral cluster contingency table saved in {filepath}\n')

# save KMeans and Spectral labels
cluster_labels = pd.DataFrame({
    'KMeans': kmeans_labels,
    'Spectral': spectral_labels
})
filepath = 'data/cluster_labels.csv'
cluster_labels.to_csv(filepath, index=False)
print(f'Cluster labels saved in {filepath}\n')

########################################

###  2D visualisation of clusters  ###

# below we obtain 2D representations of the 
# dataset using PCA and UMAP, then colour the 
# points according to the kmeans labels and spectral labels
########################################

# applying PCA to reduce the data to 2 dimensions
pca = PCA(n_components=2)
principal_components = pca.fit_transform(X)  # Excluding the 'Unnamed: 0' and 'classification' columns

# dataframe for easy plotting
pca_df = pd.DataFrame(data = principal_components, columns = ['Principal Component 1', 'Principal Component 2'])
pca_df['type'] = y
pca_df['KMeans labels'] = kmeans_labels
pca_df['Spectral labels'] = spectral_labels

# apply UMAP to reduce the data to 2 dimensions
reducer = umap.UMAP(n_components=2, random_state=0)
umap_results = reducer.fit_transform(X)

# dataframe for easy plotting
umap_df = pd.DataFrame(umap_results, columns=['UMAP 1', 'UMAP 2'])
umap_df['type'] = y
umap_df['KMeans labels'] = kmeans_labels
umap_df['Spectral labels'] = spectral_labels

# create a single figure and three subplots (axes)
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
axes = axes.flatten()

# PCA with KMeans labels
sns.scatterplot(x='Principal Component 1', y='Principal Component 2', data=pca_df, hue='KMeans labels', palette=['#ff7f0e', '#1f77b4'], ax=axes[0])

# PCA with spectral labels
sns.scatterplot(x='Principal Component 1', y='Principal Component 2', data=pca_df, hue='Spectral labels', palette=['#1f77b4', '#ff7f0e'], ax=axes[1])

# UMAP with KMeans labels
sns.scatterplot(x='UMAP 1', y='UMAP 2', data=umap_df, hue='KMeans labels', palette=['#ff7f0e', '#1f77b4'], ax=axes[2])

# UMAP with spectral
sns.scatterplot(x='UMAP 1', y='UMAP 2', data=umap_df, hue='Spectral labels', palette=['#1f77b4', '#ff7f0e'], ax=axes[3])

# text labels
axes[0].text(1.8, 2, '(a)', fontsize=20, color='black') 
axes[1].text(1.8, 2, '(b)', fontsize=20, color='black') 
axes[2].text(7.2, 8.8, '(c)', fontsize=20, color='black') 
axes[3].text(7.2, 8.8, '(d)', fontsize=20, color='black') 

for ax in axes:
    ax.grid(True)

filepath = 'plots/5a_cluster_visualisation.png'
fig.savefig(filepath)
print(f'2D visualisation of KMeans and Spectral cluster saved in {filepath}')