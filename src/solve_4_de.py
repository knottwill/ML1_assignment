import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os
import warnings
from utils import classifier_evaluation_plot

# ignore FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# make plots/ directory if it doesn't already exist
if not os.path.exists('plots/'):
    os.makedirs('plots/')

# load pre-processed dataset
data = pd.read_csv('data/4_preprocessed.csv')
X = data.drop(columns=['Unnamed: 0', 'type'])
y = data['type']
seed = 0

# splitting the data into training, validation and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)

# splitting the training data into training and validation sets 
X_train, X_validate, y_train, y_validate = train_test_split(X_train, y_train, test_size=0.2, random_state=seed, stratify=y_train)

# refining a range of values for n_estimators
n_estimators_values = np.array([10, 50, 100, 200, 500, 1000, 2000, 3000])

# training and evaluating a model for each value of n_estimators
accuracy_scores = []
for n_estimators in n_estimators_values:

    # train classifier 
    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=seed)
    rf.fit(X_train, y_train)

    # evaluate accuracy on validation set
    y_pred = rf.predict(X_validate)
    accuracy_scores.append(accuracy_score(y_validate, y_pred))

# choose n_estimators with the highest accuracy on the validation set
best_n_estimators = n_estimators_values[np.argmax(accuracy_scores)]
print(f"Optimal number of trees: {best_n_estimators}")

# combining training and validation sets (to be the full training set for the optimised classifier)
X_train, y_train = pd.concat((X_train, X_validate)), pd.concat((y_train, y_validate))

# train classifier with optimised n_estimators parameter
rf = RandomForestClassifier(n_estimators=best_n_estimators, random_state=seed)
rf.fit(X_train, y_train)

# -------------------
# evaluating the classifier
# ----------------

y_pred = rf.predict(X_test)

filepath = 'plots/4d_optimised_rf.png'
classifier_evaluation_plot(y_test, y_pred, rf.classes_, filepath)
print(f'\nOptimised random forest performance on test set saved in {filepath}\n')

# ---------------------
# Feature importance
# ---------------------

# feature importances dataframe
importances= pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf.feature_importances_
})
# sort features by importance
importances = importances.sort_values(by="Importance", ascending=False)
print('Four most important features:')
print(importances.set_index('Feature').head(4))

# plotting histogram of feature importances
fig, ax = plt.subplots(figsize=(6,6))
sns.histplot(importances['Importance'], ax=ax)
ax.set_xlabel('Feature Importance', fontsize=12)
ax.set_ylabel('Count', fontsize=12)

filepath = 'plots/4e_importance_hist.png'
fig.savefig(filepath)
print(f'\nFeature importance histogram saved in {filepath}\n')

# selecting 4 most important features
important_features = importances['Feature'][:4]

# subsetting the training and test data to include only these important features
X_train_important = X_train[important_features]
X_test_important = X_test[important_features]

# re-training the Random Forest Classifier using only the important features
rf_important = RandomForestClassifier(n_estimators=200, random_state=seed)
rf_important.fit(X_train_important, y_train)

# ------------------------
# Evaluating classifier with important features
# ------------------------

# predicting on the test set with the important features
y_pred_important = rf_important.predict(X_test_important)

filepath = 'plots/4e_importance_rf.png'
classifier_evaluation_plot(y_test, y_pred_important, rf_important.classes_, filepath)
print('Random forest trained on subset of 50 most important features.')
print(f'Performance on test set saved in {filepath}')

