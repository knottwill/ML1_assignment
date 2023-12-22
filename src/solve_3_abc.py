import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import warnings
from scipy.stats import ks_2samp

# custom imputation function using random forest
from utils import random_forest_imputation

# ignore FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# make plots/ directory if it doesn't exist
if not os.path.exists('plots/'):
    os.makedirs('plots/')

# make data/ directory if it doesn't exist
if not os.path.exists('data/'):
    os.makedirs('data/')

# load dataset C
data = pd.read_csv('data/C_MissingFeatures.csv')
X = data.drop(columns=['Unnamed: 0', 'classification'], axis=1)
y = data['classification']

# summarize missing data
missing_data = data.isnull().sum()
affected_features = list(missing_data[missing_data > 0].index)
print('Features with missing values:')
print(missing_data[missing_data > 0])

# performing random forest imputation on the dataset
print('\nBeginning imputation...')
X_imputed = random_forest_imputation(X, {'random_state': 42})
X_imputed = pd.DataFrame(X_imputed, columns = X.columns)

# save imptued dataset
data_imputed = pd.concat((data['Unnamed: 0'], X_imputed, y), axis=1)
filepath = 'data/C_imputed.csv'
data_imputed.to_csv(filepath, index=False)
print(f'Dataset imputed and saved in {filepath}\n')

# perform KS test on all affected features
p_vals = []
for feature in affected_features:

    statistic, p_value = ks_2samp(X[feature].dropna(), X_imputed[feature])
    p_vals.append(p_value)

assert min(p_vals) > 0.05
print("KS tests performed on all imputed features.")
print(f"Minimum p-value from all KS tests: {min(p_vals)}")

# plotting original and imputed distributions of the first 4 affected features
fig, axes = plt.subplots(3, 4, figsize=(12, 9))
axes = axes.flatten()

for i, feature in enumerate(affected_features):

    # imputed Distribution
    sns.histplot(X_imputed[feature], bins=30, color="orange", label="Imputation", ax=axes[i], alpha=1)

    # original distribution
    sns.histplot(X[feature], bins=30, color="green", label="Original", ax=axes[i], alpha=1)
    
    axes[i].set_title(feature)
    axes[i].set_xlabel('value')
    axes[i].set_ylabel('Count')
    axes[i].legend()

plt.tight_layout()

filepath = 'plots/3c_imputation.png'
fig.savefig(filepath)
print(f'\nOriginal vs Imputed comparison plot saved in {filepath}')