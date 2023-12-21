import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from utils import classifier_evaluation_plot

# make plots/ directory if it doesn't already exist
if not os.path.exists('plots/'):
    os.makedirs('plots/')

# load pre-processed dataset
data = pd.read_csv('data/4_preprocessed.csv')
X = data.drop(columns=['Unnamed: 0', 'type'])
y = data['type']
seed = 42

# splitting the data into 80% training and 20% testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)

# train classifier with default parameters
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

# predict on the test set
y_pred = rf.predict(X_test)

# -------------------
# evaluating the classifier
# ----------------

filepath = 'plots/4c.png'
classifier_evaluation_plot(y_test, y_pred, rf.classes_, filepath)
print(f'Results saved in {filepath}')