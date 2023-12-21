import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import warnings
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from scipy.stats import ks_2samp

# import FutureWarnings from sns.histplot
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
print(f"Removed {len(outlier_features)} features")

# remaining outlier values
outlier_values = X_outliers_removed.isna()

print("After removing features containing only zeros and outlier values,")
print(f"there are {X_outliers_removed.isna().sum().sum()} outlier values left to correct.")

n_outliers = outlier_values.sum().sum()
features_with_outliers = outlier_values.any()[outlier_values.any()].index
n_samples_with_outliers = np.count_nonzero(outlier_values.any(axis=1))
print(f"Found {n_outliers} outlier values, across {len(features_with_outliers)} features and {n_samples_with_outliers} samples")

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

features_to_compare = ['Fea336', 'Fea493', 'Fea71', 'Fea395']
for i, feature in enumerate(features_to_compare):

    # Combine the data to find common range
    combined_data = np.concatenate([X[feature].dropna(), X_outliers_removed[feature].dropna()])
    
    # # Define the bins
    bins = np.linspace(combined_data.min(), combined_data.max(), 31)

    # Original Distribution
    sns.histplot(X[feature], bins=bins, color="blue", label="Original", ax=axes[0, i], alpha=0.5)

    # Imputed Distribution
    sns.histplot(X_imputed[feature], bins=bins, color="green", label="Imputed", ax=axes[1, i], alpha=0.5)

    # original Distribution
    # sns.histplot(X[feature], kde=True, bins=30, color="blue", label="Original", ax=axes[0, i])
    axes[0, i].set_title(f'Original Distribution of {feature}')
    axes[0, i].legend()

    # imputed Distribution
    # sns.histplot(X_imputed[feature], kde=True, bins=30, color="green", label="Imputed", ax=axes[1, i])
    axes[1, i].set_title(f'Imputed Distribution of {feature}')
    axes[1, i].legend()

plt.tight_layout()

filepath = 'plots/3e.png'
fig.savefig(filepath)
print(f"Comparison plot saved in {filepath}")