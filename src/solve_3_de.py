import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from scipy.stats import ks_2samp
import os
import warnings

# ignore FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# make plots/ directory if it doesn't exist
if not os.path.exists('plots/'):
    os.makedirs('plots/')
output_dir = 'plots/'

# load imputed dataset
data = pd.read_csv('data/C_Imputed.csv')
X = data.drop(columns=['Unnamed: 0', 'classification'], axis=1)
y = data['classification']

# assert there are no missing values 
assert X.isna().sum().sum() == 0

# Standardizing the data
scaler = StandardScaler()
X_std = scaler.fit_transform(X)
X_std = pd.DataFrame(X_std, columns=X.columns)

# ----------------------
# Calculate sparsity of features
# ----------------------
def calc_sparsity(feature):
    return (feature==0).sum() / len(feature)

sparsity = X.apply(calc_sparsity)

# ----------------------
# Plotting Z score vs Sparsity:
# We plot the Z-score of all non-zero values of all features against 
# the sparsity of the feature it was taken from. This is done as a scatter
# plot overlaid with a 2D density histogram plot.
# ----------------------

# Initialize lists to store z-scores and corresponding sparsity values
z_scores = []
sparsity_values = []

# find z score and sparsity for all non-zero feature values in the 
# dataset, but only for features with greater than 50% sparsity
sparse_features = sparsity[sparsity > 0.5].index
for feature in sparse_features:

    # find z scores of non-zero feature values
    z = X_std[feature][X[feature] != 0]

    # add z scores and sparsity values to the respective lists
    z_scores.extend(z)
    sparsity_values.extend([sparsity[feature] for _ in range(len(z))])

fig, ax = plt.subplots(figsize=(8, 6))

# scatter plot to show the outlier points
ax.scatter(x=sparsity_values, y=z_scores, color='black', alpha=1, s=5)

# histogram with threshold to show the main 
sns.histplot(x=sparsity_values, y=z_scores, bins=50, stat='density', cbar=True, thresh=0.07, cmap='viridis', ax=ax)
ax.axhline(0, color='red', linestyle='--', linewidth=1) # line at origin
ax.set_yticks(np.arange(-7, 20, 2))
ax.set_xlabel('Sparsity')
ax.set_ylabel('Z-score')
ax.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()

filepath = 'plots/3d_Z_vs_Sparsity.png'
fig.savefig(filepath)
print(f'\nZ score vs sparsity plot saved in {filepath}')

# ----------------------
# Find outlier values
# ----------------------

def find_outliers(feature):
    sparsity = (feature==0).sum() / len(feature)
    feature_std = (feature - feature.mean())/feature.std()

    if sparsity <= 0.2:
        return np.abs(feature_std) > 3
    
    # else
    nonzero = feature[feature != 0]
    nonzero_std = (nonzero - nonzero.mean())/nonzero.std()
    return (np.abs(feature_std) > 7) | (np.abs(nonzero_std) > 3)

# find outlier values
outlier_values = X.apply(find_outliers)

# make outlier values NaN
X_outliers_removed = X[outlier_values == False]

# -----------------------
# Remove features which contain only zeros and outlier values
# -----------------------

outlier_features = X_outliers_removed.sum()[X_outliers_removed.sum() == 0].index
X_outliers_removed = X_outliers_removed.drop(outlier_features, axis=1)
print(f"\nRemoved {len(outlier_features)} features containing only zeros and outlier values.")

# remaining outlier values
outlier_values = X_outliers_removed.isna()
n_outliers = outlier_values.sum().sum()
features_with_outliers = outlier_values.any()[outlier_values.any()].index
n_samples_with_outliers = np.count_nonzero(outlier_values.any(axis=1))
print(f"Out of the remaining features:")
print(f"We found {n_outliers} outlier values, across {len(features_with_outliers)} features and {n_samples_with_outliers} samples\n")

# --------------------
# Imputing outlier values
# --------------------

imputer = KNNImputer(n_neighbors=10, weights='distance')
X_imputed = imputer.fit_transform(X_outliers_removed)
X_imputed = pd.DataFrame(X_imputed, columns = X_outliers_removed.columns)

# ----------------------
# KS test for original and imputed features
# ----------------------

p_vals = []
for feature in features_with_outliers:

    statistic, p_value = ks_2samp(X[feature], X_imputed[feature])

    p_vals.append(p_value)

print("We performed KS tests on all imputed features.")
print(f"The minimum p-value yielded was {min(p_vals)}")
assert min(p_vals) > 0.05

# --------------------
# Comparing four imputed features
# --------------------

# plotting original and imputed distributions of the first 4 affected features
fig, axes = plt.subplots(2, 4, figsize=(15, 10))

features_to_compare = ['Fea336', 'Fea71', 'Fea395', 'Fea493']
for i, feature in enumerate(features_to_compare):

    # Combine the data to find common range to define the bins
    combined_data = np.concatenate([X[feature], X_outliers_removed[feature].dropna()])
    bins = np.linspace(combined_data.min(), combined_data.max(), 31)

    # full original distribution
    sns.histplot(X[feature], bins=bins, color="orange", label="Outliers", ax=axes[0, i], alpha=0.7)

    # original distribution with outliers removed (overlaid to highlight outliers)
    sns.histplot(X_outliers_removed[feature], bins=bins, color="blue", ax=axes[0, i], alpha=0.5)

    axes[0, i].set_title(f'Original Distribution of {feature}')
    axes[0, i].legend()
    axes[0, i].set_xlabel('value')

    # imputed Distribution
    sns.histplot(X_imputed[feature], bins=bins, color="orange", label="Imputation", ax=axes[1, i], alpha=0.7)

    # original distribution with outliers removed (overlaid to highlight imputations)
    sns.histplot(X_outliers_removed[feature], bins=bins, color="green", ax=axes[1, i], alpha=1)

    axes[1, i].set_title(f'Imputed Distribution of {feature}')
    axes[1, i].legend()
    axes[1, i].set_xlabel('value')

plt.tight_layout()

filepath = 'plots/3e_comparison.png'
fig.savefig(filepath)
print(f"\nOriginal vs Imputation comparison plot saved in {filepath}")