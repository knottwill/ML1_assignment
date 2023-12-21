import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import warnings
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from utils import random_forest_imputation
from scipy.stats import ks_2samp

# import FutureWarnings from sns.histplot
warnings.simplefilter(action='ignore', category=FutureWarning)

# make plots/ directory if it doesn't exist
if not os.path.exists('plots/'):
    os.makedirs('plots/')
output_dir = 'plots/'

# make data/ directory if it doesn't exist
if not os.path.exists('data/'):
    os.makedirs('data/')

data = pd.read_csv('data/C_MissingFeatures.csv')
X = data.drop(columns=['Unnamed: 0', 'classification'], axis=1)
y = data['classification']

# summarize missing data
missing_data = data.isnull().sum()
affected_features = list(missing_data[missing_data > 0].index)
print('Features with missing values:')
print(missing_data[missing_data > 0])

# performing random forest imputation on the dataset
X_imputed = random_forest_imputation(X, {'random_state': 42})
X_imputed = pd.DataFrame(X_imputed, columns = X.columns)

# save imptued dataset
data_imputed = pd.concat((data['Unnamed: 0'], X_imputed, y), axis=1)
data_imputed.to_csv('data/C_imputed.csv', index=False)
print('\nDataset with missing values imputed saved in data/C_imputed.csv')

# Perform KS test on all affected features
p_vals = []
for feature in affected_features:

    statistic, p_value = ks_2samp(X[feature].dropna(), X_imputed[feature])

    p_vals.append(p_value)

assert min(p_vals) > 0.05
print(f"Minimum p-value from all KS tests: {min(p_vals)}")
print("This is indicates that the original and imputed distributions are drawn from the same distribution")
print("This shows that the imputation was successful in preserving the statisitcal properties of the original distributions")

# plotting original and imputed distributions of the first 4 affected features
fig, axes = plt.subplots(3, 4, figsize=(12, 9))
axes = axes.flatten()

for i, feature in enumerate(affected_features):

    # combine the data to find common range
    combined_data = np.concatenate([X_imputed[feature].dropna(), X[feature].dropna()])
    
    # define the bins
    bins = np.linspace(combined_data.min(), combined_data.max(), 31)

    # imputed Distribution
    sns.histplot(X_imputed[feature], bins=30, color="orange", label="Imputation", ax=axes[i], alpha=1)

    # original distribution
    sns.histplot(X[feature], bins=30, color="green", label="Original", ax=axes[i], alpha=1)
    axes[i].set_title(feature)
    axes[i].set_xlabel('value')
    axes[i].set_ylabel('Count')
    axes[i].legend()

plt.tight_layout()

filepath = output_dir + '3c.png'
fig.savefig(filepath)
print(f'\nOriginal vs Imputed comparison plot saved in {filepath}')