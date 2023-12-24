import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
import os
import warnings
warnings.filterwarnings('ignore')


if not os.path.exists('plots/'):
    os.makedirs('plots/')

# load preprocessed dataset
data = pd.read_csv('data/5_preprocessed.csv')
X = data.drop(['Unnamed: 0','type'], axis=1)
y = data['type']
seed = 42

# load kmeans and spectral cluster labels from part (a)
cluster_labels = pd.read_csv('data/cluster_labels.csv')
kmeans_labels = cluster_labels["KMeans"]
spectral_labels = cluster_labels["Spectral"]

# -----------------------
# Predicting KMeans clusters
# -----------------------

# Splitting the data into training and test sets (80% train, 20% test)
X_train, X_test, km_train, km_test = train_test_split(X, kmeans_labels, test_size=0.2, random_state=seed, stratify=kmeans_labels)

log_reg = LogisticRegression(penalty='l1', solver= 'saga', max_iter=10000, random_state=seed)

# Define the parameter grid for GridSearchCV
param_grid = {
    'C': [0.01, 0.1, 1, 10],  # Regularization strengths
}

grid_search = GridSearchCV(estimator=log_reg, param_grid=param_grid, cv=5, n_jobs=-1)

# Train the classifier with GridSearchCV
grid_search.fit(X_train, km_train)

# Use the best estimator found by GridSearchCV
best_log_reg = grid_search.best_estimator_

# Predict on the test set with the best estimator
km_pred = best_log_reg.predict(X_test)

# -----------------------
# Predicting Spectral clusters
# -----------------------

# Splitting the data into training and test sets (80% train, 20% test)
X_train, X_test, sp_train, sp_test = train_test_split(X, spectral_labels, test_size=0.2, random_state=seed, stratify=spectral_labels)

rf = RandomForestClassifier(random_state=seed)

# Define the parameter grid for GridSearchCV
param_grid = {
    'n_estimators': [10, 50, 100, 200],  # Number of trees in the forest
    'max_depth': [None, 10, 20, 30],  # Maximum depth of the tree
    # Add other parameters as needed
}

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1)

# Train the classifier with GridSearchCV
grid_search.fit(X_train, sp_train)

# Use the best estimator found by GridSearchCV
best_rf = grid_search.best_estimator_

# Predict on the test set with the best estimator
sp_pred = best_rf.predict(X_test)

# ------------------
# Confusion matrices of classifiers
# ------------------

# Confusion Matrices
conf_matrix_km = confusion_matrix(km_test, km_pred)
conf_matrix_sp = confusion_matrix(sp_test, sp_pred)

# Create a figure with two subplots, side by side
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

# Plot the confusion matrix for KMeans
ConfusionMatrixDisplay(conf_matrix_km, display_labels=best_log_reg.classes_).plot(ax=axes[0])
axes[0].set_xlabel('Predicted KMeans labels', fontsize=14)
axes[0].set_ylabel('True KMeans labels', fontsize=14)
axes[0].text(-0.1, 1.1, '(a)', transform=axes[0].transAxes, fontsize=16, fontweight='bold')

# Plot the confusion matrix for Spectral Clustering
ConfusionMatrixDisplay(conf_matrix_sp, display_labels=best_rf.classes_).plot(ax=axes[1])
axes[1].set_xlabel('Predicted Spectral labels', fontsize=14)
axes[1].set_ylabel('True Spectral labels', fontsize=14)
axes[1].text(-0.1, 1.1, '(b)', transform=axes[1].transAxes, fontsize=16, fontweight='bold')

# Adjust layout
plt.subplots_adjust(wspace=10)
plt.tight_layout()

filepath = 'plots/5b_confusion.png'
fig.savefig(filepath)
print(f'\nConfusion matrices of classifiers saved in {filepath}\n')

# -------------------------
# Calculating feature importances for 
# predicting KMeans and Spectral clusterings
# -------------------------

# feature importances for KMeans label prediction
# (using the absolute coefficients from logistic regression)
importances_km = pd.DataFrame({
    'Feature': X.columns,
    'Importance': np.abs(best_log_reg.coef_[0])
})

# sort features by importance
importances_km = importances_km.sort_values(by="Importance", ascending=False)

# take 50 most important features (as long as they have non-zero importance)
important_features_km = importances_km[:50]
assert (important_features_km['Importance'] > 0).all()

# first and second most discriminant features
most_discriminant_km = important_features_km.iloc[0]['Feature']
second_discriminant_km = important_features_km.iloc[1]['Feature']

# feature importances for Spectral label prediction
importances_sp = pd.DataFrame({
    'Feature': X.columns,
    'Importance': best_rf.feature_importances_
})
# sort features by importance
importances_sp = importances_sp.sort_values(by="Importance", ascending=False)

# take 50 most important features (as long as they have non-zero importance)
important_features_sp = importances_sp[:50]
assert (important_features_sp['Importance'] > 0).all()

# first and second most discriminant features
most_discriminant_sp = important_features_sp.iloc[0]['Feature']
second_discriminant_sp = important_features_sp.iloc[1]['Feature']

# ---------------
# Repeating clustering on just the important features
# ---------------

X_important_km = X[important_features_km['Feature']]

X_important_sp = X[important_features_sp['Feature']]

n_clusters = 2
km = KMeans(n_clusters=n_clusters, random_state=seed)
kmeans_labels_important = km.fit_predict(X_important_km)

spectral = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', random_state=seed)
spectral_labels_important = spectral.fit_predict(X_important_sp)

# ------------------
# Contingency tables of clusterings
# before and after removing unimportant features.
# ------------------

# contingency tables
contingency_km = confusion_matrix(kmeans_labels, kmeans_labels_important)
contingency_sp = confusion_matrix(spectral_labels, spectral_labels_important)

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

# contingency table for KMeans
ConfusionMatrixDisplay(contingency_km, display_labels=best_log_reg.classes_).plot(ax=axes[0], cmap="YlGn")
axes[0].set_xlabel('KMeans on important features', fontsize=14)
axes[0].set_ylabel('KMeans on full dataset', fontsize=14)
axes[0].text(-0.1, 1.1, '(a)', transform=axes[0].transAxes, fontsize=16, fontweight='bold')

# contingency table for Spectral Clustering
ConfusionMatrixDisplay(contingency_sp, display_labels=best_rf.classes_).plot(ax=axes[1], cmap="YlGn")
axes[1].set_xlabel('Spectral clustering on important features', fontsize=14)
axes[1].set_ylabel('Spectral clustering on full dataset', fontsize=14)
axes[1].text(-0.1, 1.1, '(b)', transform=axes[1].transAxes, fontsize=16, fontweight='bold')

plt.subplots_adjust(wspace=10)
plt.tight_layout()

filepath = 'plots/5b_contingency.png'
fig.savefig(filepath)
print(f'Contingency tables of clusterings saved in {filepath}\n')

# --------------------
# Plotting PCA visualisation with points coloured by:
# (i) KMeans cluster labels
# (ii) Most discriminant feature for KMeans label prediction
# (iii) Second most discriminant feature
# --------------------------------

# apply PCA to reduce the data to 2 dimensions
pca = PCA(n_components=2)
principal_components = pca.fit_transform(X)  # Excluding the 'Unnamed: 0' and 'classification' columns

# dataFrame for easy plotting
pca_df = pd.DataFrame(data = principal_components, columns = ['Principal Component 1', 'Principal Component 2'])
pca_df['type'] = y
pca_df['KMeans labels'] = kmeans_labels
pca_df['Spectral labels'] = spectral_labels
pca_df['1st discriminant feature'] = X[most_discriminant_km]
pca_df['2nd discriminant feature'] = X[second_discriminant_km]

fig, axes = plt.subplots(3, 1, figsize=(8, 20))

# PCA scatter plot coloured by KMeans labels
sns.scatterplot(x='Principal Component 1', y='Principal Component 2', data=pca_df, 
                hue='KMeans labels', palette=['#ff7f0e', '#1f77b4'], ax=axes[0])

# same plot, coloured by most discriminant feature
scatter_2 = sns.scatterplot(x='Principal Component 1', y='Principal Component 2', data=pca_df, 
                            hue='1st discriminant feature', palette='viridis', ax=axes[1])
scatter_2.legend_.remove()
cbar_2 = plt.colorbar(scatter_2.collections[0], ax=axes[1], orientation='vertical')
cbar_2.set_label('Most Discriminant Feature', fontsize=10)

# same plot, coloured by second most discriminant feature
scatter_3 = sns.scatterplot(x='Principal Component 1', y='Principal Component 2', data=pca_df, 
                            hue='2nd discriminant feature', palette='viridis', ax=axes[2])
scatter_3.legend_.remove()
cbar_3 = plt.colorbar(scatter_3.collections[0], ax=axes[2], orientation='vertical')
cbar_3.set_label('Second Most Discriminant Feature', fontsize=10)

for i, label in enumerate(['(i)', '(ii)', '(iii)']):
    axes[i].grid(True)
    axes[i].set_xlabel('Principal Component 1', fontsize=14)
    axes[i].set_ylabel('Principal Component 2', fontsize=14)
    axes[i].text(-2, 2, label, fontsize=16, fontweight='bold')

filepath = 'plots/5c.png'
fig.savefig(filepath)
print(f'PCA visualisation saved in {filepath}')
