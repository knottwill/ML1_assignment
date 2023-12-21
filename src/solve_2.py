import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from utils import classifier_evaluation_plot
import numpy as np
import pandas as pd

data = pd.read_csv('data/B_Relabelled.csv')
X = data.drop(['Unnamed: 0', 'classification'], axis=1)
features = list(X.columns)
y = data['classification']

# print label frequencies
label_frequency = data['classification'].value_counts(dropna=False).reset_index()
print(label_frequency)

# assert there are no missing features (only missing labels)
assert X.isnull().sum().sum() == 0

# ------------------------
# Finding and addressing duplicates
# ------------------------

def process_duplicates(duplicates):
    """
    Take pair of duplicates and return the instance to keep
    (If classification labels agree, keep instance with that label.
    If classification labels disagree, keep instance with missing label.)
    """

    # asserting that there are exactly 2 duplicates
    assert len(duplicates) == 2
    
    # get instance of duplicated sample
    instance = duplicates.iloc[0]

    # if classifications differ, make the classification NaN
    if duplicates['classification'].nunique() != 1:
        instance['classification'] = np.NaN
        
    return instance

def inconsistent_labels(duplicates):
    return duplicates['classification'].nunique() > 1

# find duplicates and group them together
duplicates_df = data[data.duplicated(subset=features, keep=False)]
grouped_duplicates = duplicates_df.groupby(features, as_index=False)
print(f"\n{len(grouped_duplicates)} pairs of duplicates found")

# find number of duplicates with inconsistent labels
inconsistent_duplicates = grouped_duplicates.filter(inconsistent_labels)
inconsistent_duplicates = inconsistent_duplicates.groupby(features, as_index=False)
print(f"{len(inconsistent_duplicates)} out of {len(grouped_duplicates)} pairs have inconsistent labels\n")

# process duplicates
processed_duplicates = grouped_duplicates.apply(process_duplicates)

# drop original duplicates from the dataframe and append the processed ones
data_deduplicated = data.drop_duplicates(subset=features, keep=False)
data = pd.concat([data_deduplicated, processed_duplicates], ignore_index=True)

# assert there are no more duplicates
assert np.count_nonzero( data.duplicated(subset=features, keep=False) ) == 0

# print label frequencies after correcting duplicates
print(data['classification'].value_counts(dropna=False).reset_index())

# ---------------------
# k-NN Classifier
# ---------------------

# get the subset of data without missing labels
data_no_missing = data.dropna(subset=['classification'])
X = data_no_missing.drop(['Unnamed: 0', 'classification'], axis=1)
y = data_no_missing['classification']
X, y = np.array(X), np.array(y)

# split into stratified train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# initialize KNN and GridSearchCV
knn = KNeighborsClassifier()
param_grid = {
    'n_neighbors': np.arange(1, 31), 
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

# perform grid search cross validation to select best hyperparameters
grid_search = GridSearchCV(knn, param_grid, cv=5)
grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_

# train KNN with the best hyperparameters
knn = KNeighborsClassifier(**best_params)
knn.fit(X_train, y_train)

# evaluation k-NN classifier on test set
y_pred = knn.predict(X_test)

filepath = 'plots/2.png'
classifier_evaluation_plot(y_test, y_pred, knn.classes_, filepath)
print(f'Results saved in {filepath}')

# ---------------------
# Predicting missing labels
# ---------------------

# get the subset of data with missing labels
data_missing_labels = data[data['classification'].isna()]
X_missing = data_missing_labels.drop(['Unnamed: 0', 'classification'], axis=1)
X_missing = np.array(X_missing)

# predict and fill in missing labels
predicted_labels = knn.predict(X_missing)
data_missing_labels.loc[:, 'classification'] = predicted_labels

# combine the dataset with missing labels now filled and the dataset with existing labels
processed_data = pd.concat([data_no_missing, data_missing_labels], ignore_index=True)

print(processed_data['classification'].value_counts(dropna=False).reset_index())
