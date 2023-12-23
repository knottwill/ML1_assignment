import numpy as np
import pandas as pd
import os
from sklearn.utils import resample
from sklearn.preprocessing import MinMaxScaler

from utils import random_forest_imputation

# make data/ directory if it doesn't exist
if not os.path.exists('data/'):
    os.makedirs('data/')

# load dataset
data = pd.read_csv('data/ADS_baselineDataset.csv')
X = data.drop(['Unnamed: 0', 'type'], axis=1)
y = data['type']
seed = 42

# assert there are no missing values
assert np.count_nonzero(data.isnull()) == 0

# -----------------------------
# Calculating sparsity of features
# -----------------------------

def calc_sparsity(feature):
    return (feature==0).sum() / len(feature)

sparsity = X.apply(calc_sparsity)

# calculating percentage of features that are at various levels of sparsity
print(f"\n{(sparsity == 1).sum()*100 / len(sparsity)}% of features contain only 0 values (100% sparsity)")
print(f"{(sparsity >= 495/500).sum()*100 / len(sparsity)}% of features have 5 or less non-zero instances (>= 99% sparsity)")
print(f"{(sparsity >= 475/500).sum()*100 / len(sparsity)}% of features have 25 or less non-zero instances (>= 95% sparsity)")
print(f"{(sparsity >= 450/500).sum()*100 / len(sparsity)}% of features have 50 or less non-zero instances (>= 90% sparsity)\n")

# -------------------------------
# Removing features with sparsity >= 95%
# (since they increase dimensionality and introduce noise 
# without contributing useful information)
# --------------------------------

# remove highly sparse features
highly_sparse_features = sparsity[sparsity >= 0.95].index
X = X.drop(columns=highly_sparse_features)

# re-calculate sparsity
sparsity = X.apply(calc_sparsity)

print(f"{len(highly_sparse_features)} features had sparsity >= 95% and were removed from the dataset\n")
print(f"#### Out of the remaining features: ####")
print(f'++++ {round((sparsity >= 250/500).sum()*100/ len(sparsity),1)}% of features have sparsity >= 50%')


# assert that the dtypes of all features is np.float64
assert len(set(X.dtypes)) == 1
assert list(set(X.dtypes))[0] == np.float64

# -----------------------
# Check there are no (more) near-zero variance features
# variance threshold is 0.002 after normalisation
# -----------------------

# normalise features
norm = MinMaxScaler()
normalised_features = norm.fit_transform(X)
normalised_features = pd.DataFrame(normalised_features, columns=X.columns)

# calculate variances for each feature
variances = normalised_features.var()

# assert there are no near zero variance features 
# (variance of less than 0.002 after normalisation)
if np.count_nonzero(variances < 0.002) == 0:
    print('++++ No Near-Zero Variance features found')
else:
    raise ValueError('Near-Zero Variance features found\n')

# --------------------------------
# Check there is no multi-collinearity
# (no pairs of features with an absolute correlation 
# coefficient larger than 0.9)
# --------------------------------

# Assuming 'df' is your DataFrame
correlation_matrix = X.corr()

# get rid of the diagonal values (since these are all 1)
np.fill_diagonal(correlation_matrix.values, np.nan)

# find the maximum absolute correlation value (excluding diagonal)
max_corr = correlation_matrix.abs().max().max()

if max_corr < 0.9:
    print("++++ No features are highly correlated")
    print(f"++++ (Maximum absolute correlation between any two features: {max_corr: .3})\n")
else:
    raise ValueError(f"Highly correlated features found")

# -------------------------
# Correcting extreme outlier values
# -------------------------

def find_extreme_outliers(feature):
    """
    Identify extreme outlier values within a given feature.

    This function detects extreme outliers in a feature by calculating the z-scores, 
    which represent the number of standard deviations a data point is from the mean. 
    The method varies depending on the sparsity of the feature (the proportion of zero values):

    1. For non-sparse features (sparsity <= 20%):
       - An outlier is identified if its z-score (considering all values) is greater than 5.

    2. For sparse features (sparsity > 20%), a value is considered an outlier if either of
       the following conditions are satisfied:
       a. Its z-score, calculated using all values (including zeros), is greater than 10.
       b. It is a non-zero value and its 'secondary' z-score, calculated using only non-zero values,
          is greater than 3. (Ie. if it deviates from the mean of the non-zero values by 5
          standard deviations)

    Parameters
    -----------
    feature (pd.Series): A single feature (column) from a DataFrame.

    Returns
    -----------
    pd.Series: A boolean Series where True indicates an outlier value in the input feature.
    """

    # calculate sparsity of feature
    sparsity = (feature==0).sum() / len(feature)

    # standardise feature (ie. calculate z-scores)
    feature_std = (feature - feature.mean())/feature.std()

    # outlier if sparsity <= 20% and z-score > 5
    if sparsity <= 0.2:
        return np.abs(feature_std) > 5
    
    # calculate secondary z-score (calculated using only non-zero values)
    nonzero = feature[feature != 0]
    nonzero_std = (nonzero - nonzero.mean())/nonzero.std()

    # outlier if sparsity > 20% and z-score > 10 or
    # secondary z-score > 5
    return (np.abs(feature_std) > 10) | (np.abs(nonzero_std) > 5)

# find (extreme) outlier values
outlier_vals = X.apply(find_extreme_outliers)

# set outlier values to NaN
X_outliers_removed = X[outlier_vals == False]

# impute outliers using random forest
print(f'Beginning imputation of {(outlier_vals).sum().sum()} outlier values...')
X_imputed = random_forest_imputation(X_outliers_removed, {'random_state': seed})
X_imputed = pd.DataFrame(X_imputed, columns = X_outliers_removed.columns)
print('finished.\n')

# -------------------
# Resampling to make class frequency uniform
# -------------------

# printing label frequency summary
label_frequency = y.value_counts(dropna=False).reset_index()
print("Frequency of labels:")
print(label_frequency)

data = pd.concat([data['Unnamed: 0'], X_imputed, data['type']], axis=1)

# separate the dataset into class subsets
df_class_1 = data[data['type'] == 1]
df_class_2 = data[data['type'] == 2]
df_class_3 = data[data['type'] == 3]

# size of biggest class
max_size = df_class_1.shape[0]

# over-sample other classes to match the max size
df_class_2_oversampled = resample(df_class_2, replace=True, n_samples=max_size, random_state=seed)
df_class_3_oversampled = resample(df_class_3, replace=True, n_samples=max_size, random_state=seed)
print(f"\nWe oversampled minority classes to match size of class 1 ({max_size})")

# combine the oversampled datasets
oversampled_data = pd.concat([df_class_1, df_class_2_oversampled, df_class_3_oversampled])

# shuffle the dataset
oversampled_data = oversampled_data.sample(frac=1, random_state=seed).reset_index(drop=True)
data = oversampled_data

filepath = 'data/4_preprocessed.csv'
data.to_csv(filepath, index=False)
print(f'\nPre-processed data saved in {filepath}')