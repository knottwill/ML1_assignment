import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.utils import resample
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.impute import KNNImputer
from utils import random_forest_imputation

# make data/ directory if it doesn't exist
if not os.path.exists('data/'):
    os.makedirs('data/')

# load dataset
data = pd.read_csv('data/ADS_baselineDataset.csv')
X = data.drop(['Unnamed: 0', 'type'], axis=1)
y = data['type']
seed = 42
# seed = 123

# assert there are no missing values
assert np.count_nonzero(data.isnull()) == 0

# printing label frequency summary
label_frequency = y.value_counts(dropna=False).reset_index()
print("Frequency of labels:")
print(label_frequency)

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

print(f"{len(highly_sparse_features)} features had sparsity >= 95% and were removed from the dataset")
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
    print("++++ No features are highly correlated:")
    print(f"++++ (Maximum absolute correlation between any two features: {max_corr: .3})\n")
else:
    raise ValueError(f"Highly correlated features found")

# -------------------------
# Correcting extreme outlier values
# -------------------------

def find_outliers(feature):
    sparsity = (feature==0).sum() / len(feature)
    feature_std = (feature - feature.mean())/feature.std()

    if sparsity <= 0.2:
        return np.abs(feature_std) > 5
    
    # else
    nonzero = feature[feature != 0]
    nonzero_std = (nonzero - nonzero.mean())/nonzero.std()
    return (np.abs(feature_std) > 10) | (np.abs(nonzero_std) > 5)

outlier_vals = X.apply(find_outliers)

print(f"{(outlier_vals).sum().sum()} outlier values imputed\n")

X_na = X[outlier_vals == False]

X_imputed = random_forest_imputation(X_na, {'random_state': seed})
X_imputed = pd.DataFrame(X_imputed, columns = X_na.columns)

# imputer = KNNImputer(n_neighbors=10, weights='distance')
# X_imputed = imputer.fit_transform(X_na)
# X_imputed = pd.DataFrame(X_imputed, columns = X_na.columns)

# -------------------
# Resampling to make them uniform
# -------------------

data = pd.concat([data['Unnamed: 0'], X_imputed, data['type']], axis=1)

# Separate the dataset into different class subsets
df_class_1 = data[data['type'] == 1]
df_class_2 = data[data['type'] == 2]
df_class_3 = data[data['type'] == 3]

# Find the class with the maximum samples
max_size = df_class_1.shape[0]

# Upsample other classes to match the max size
df_class_2_oversampled = resample(df_class_2, replace=True, n_samples=max_size, random_state=seed)
df_class_3_oversampled = resample(df_class_3, replace=True, n_samples=max_size, random_state=seed)

# Combine the oversampled datasets
oversampled_data = pd.concat([df_class_1, df_class_2_oversampled, df_class_3_oversampled])

# Shuffle the dataset to prevent the model from learning patterns based on the order of the rows
oversampled_data = oversampled_data.sample(frac=1, random_state=seed).reset_index(drop=True)
data = oversampled_data

data.to_csv('data/4_preprocessed.csv', index=False)