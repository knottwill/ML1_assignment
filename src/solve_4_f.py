import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from utils import classifier_evaluation_plot
import warnings

# ignore FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)

data = pd.read_csv('data/4_preprocessed.csv')
X = data.drop(['Unnamed: 0', 'type'], axis=1)
y = data['type']
seed = 0

# -----------------
# Normalise features
# -----------------

# normalise the features
norm = MinMaxScaler()
X_norm = norm.fit_transform(X)

# Convert back to DataFrame
X_norm = pd.DataFrame(X_norm, columns=X.columns)

X = X_norm

# -------------------
# Train classifier
# -------------------

# Split the data into stratified training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)

# logistic regression model with Lasso (L1) regularization
model = LogisticRegression(penalty='l1', solver='saga', max_iter=10000, random_state=seed)

# Train the model
model.fit(X_train, y_train)

# evaluate the model on the test set
y_pred = model.predict(X_test)
filepath = 'plots/4f_default_lasso.png'
classifier_evaluation_plot(y_test, y_pred, model.classes_, filepath)
print(f'\nLogistic regression performance on test set saved in {filepath}\n')

# -------------------
# Feature importances
# -------------------

# feature importances dataframe
importances= pd.DataFrame({
    'Feature': X.columns,
    'Importance': np.abs(model.coef_[0]) + np.abs(model.coef_[1]) + np.abs(model.coef_[2])
})
# sort features by importance
importances = importances.sort_values(by="Importance", ascending=False)
print('Four most important features:')
print(importances.set_index('Feature').head(4))

# find non-zero importances
nonzero_importances = importances[importances['Importance'] > 0]
print(f'\nOnly {len(nonzero_importances)} features out of {len(X.columns)} have non-zero importance\n')

# plotting histogram of feature importances for the non-zero importances only
fig, ax = plt.subplots(figsize=(6,6))
sns.histplot(nonzero_importances['Importance'], bins=70, ax=ax)
ax.set_yticks([2*i for i in range(10)])
ax.set_xlabel('Feature Importance', fontsize=12)
ax.set_ylabel('Count', fontsize=12)

filepath = 'plots/4f_importance_hist.png'
fig.savefig(filepath)
print(f'Feature importance histogram saved in {filepath}')

# take 50 most important features
important_features = importances['Feature'][:4]
assert (importances['Importance'][:50] > 0).all() # assert all have non-zero importances

# logistic regression model with Lasso (L1) regularization
model = LogisticRegression(penalty='l1', solver='saga', max_iter=10000, random_state=seed)

# Train the model
model.fit(X_train[important_features], y_train)

# evaluate the model on the test set
y_pred = model.predict(X_test[important_features])
filepath = 'plots/4f_importance_lasso.png'
classifier_evaluation_plot(y_test, y_pred, model.classes_, filepath)
print('\nLogistic regression model trained on 4 most important features')
print(f'Performance on test set saved in {filepath}')