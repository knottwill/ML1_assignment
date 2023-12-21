import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from utils import classifier_evaluation_plot

data = pd.read_csv('data/4_preprocessed.csv')
X = data.drop(['Unnamed: 0', 'type'], axis=1)
y = data['type']
seed = 42

# -----------------
# Normalise features
# -----------------

# Create a MinMaxScaler object
norm = MinMaxScaler()

# Fit and transform the data
X_norm = norm.fit_transform(X)

# Convert back to DataFrame
X_norm = pd.DataFrame(X_norm, columns=X.columns)

X = X_norm

# -------------------
# Train classifier
# -------------------

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)

# Create logistic regression model with Lasso (L1) regularization
model = LogisticRegression(penalty='l1', solver='saga', max_iter=10000, random_state=seed)

# Train the model
model.fit(X_train, y_train)

# Make predictions and evaluate the model
y_pred = model.predict(X_test)
filepath = 'plots/4f_all.png'
classifier_evaluation_plot(y_test, y_pred, model.classes_, filepath)
print(f'Results saved in {filepath}')

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

# plotting histogram of feature importances
fig, ax = plt.subplots(figsize=(6,6))
sns.histplot(importances['Importance'], bins=70, ax=ax)
ax.set_xlabel('Feature Importance')

filepath = 'plots/4f_importance_hist.png'
fig.savefig(filepath)
print(f'Feature importance histogram saved in {filepath}')

# take 50 most important features
important_features = importances['Feature'][:50]
assert (importances['Importance'][:50] > 0).all() # assert all have non-zero importances

# Create logistic regression model with Lasso (L1) regularization
model = LogisticRegression(penalty='l1', solver='saga', max_iter=10000, random_state=seed)

# Train the model
model.fit(X_train[important_features], y_train)

# Make predictions and evaluate the model
y_pred = model.predict(X_test[important_features])
filepath = 'plots/4f_important.png'
classifier_evaluation_plot(y_test, y_pred, model.classes_, filepath)
print(f'Results saved in {filepath}')

# See which features are important now
importances = np.abs(model.coef_[0]) + np.abs(model.coef_[1]) + np.abs(model.coef_[2])

print(f"{(importances > 0).sum()} features were used out of {len(important_features)}")