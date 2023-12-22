"""
Module containing functions necessary for the solving script
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestRegressor

def random_forest_imputation(X, rf_params):
    """
    Random Forest Imputer

    This function takes as input a dataset X containing missing values
    then imputes them using random forest regression. When training to
    predict a given feature, the missing values in the other features 
    are temporarily imputed using the median value of that feature. 
    Returns the imputed dataset. 
    """
    X_imputed = X.copy()
    for col in X.columns:
        if X_imputed[col].isna().sum() > 0:
            # split data into sets with and without missing values
            X_with_value = X_imputed[X_imputed[col].notna()]
            X_missing = X_imputed[X_imputed[col].isna()]

            # separate predictors and target
            y_train = X_with_value[col]
            X_train = X_with_value.drop(col, axis=1)
            X_test = X_missing.drop(col, axis=1)

            # fill missing values in predictors with column median
            # (median over mean since most features are not normally distributed)
            X_train = X_train.fillna(X_train.median())
            X_test = X_test.fillna(X_train.median())

            model = RandomForestRegressor(**rf_params)

            # fit model and predict missing values
            model.fit(X_train, y_train)
            X_imputed.loc[X_missing.index, col] = model.predict(X_test)

    return X_imputed

def parse_classification_report(report):
    """
    Takes as input a 'classification report' in the style provided
    by sklearn.metrics.classification_report and parses the content
    into a dataframe. 
    """

    lines = report.split('\n')[2:-4]  # exclude the last line (accuracy)
    data = [line.split() for line in lines if line]
    headers = ["Class", "Precision", "Recall", "F1-Score", "Support"]
    return pd.DataFrame(data, columns=headers)


def classifier_evaluation_plot(y_true, y_pred, classes, filepath=None):
    """
    Function takes as input true labels, 'y_true', and predicted labels
    'y_pred', and the unique labels in the correct order 'classes' for 
    plotting, and returns a plot containing the following information:
    - Confusion matrix comparing the true and predicted labels
    - table showing the precision, recall, f1-score and support for each
      class
    """

    # calculate classification report, confusion matrix and accuracy
    class_report = classification_report(y_true, y_pred)
    conf_matrix = confusion_matrix(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)

    # convert classification report to DataFrame
    report_df = parse_classification_report(class_report)

    # plot all results
    fig, axes = plt.subplots(1, 2, figsize=(6, 3))

    # plot confusion matrix
    ConfusionMatrixDisplay(conf_matrix, display_labels=classes).plot(ax=axes[0], cmap='Greys')
    axes[0].text(-0.1, 1.1, '(a)', transform=axes[0].transAxes, fontsize=16, fontweight='bold')

    # plot the classification report as a table
    axes[1].axis('off')
    table = axes[1].table(cellText=report_df.values, colLabels=report_df.columns, loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    axes[1].text(0.2,0.2,f"Accuracy: {accuracy: .4}")
    axes[1].text(0.05, 0.96, "(b)", fontsize=16, fontweight='bold')

    plt.tight_layout()

    # save figure
    if filepath:
        plt.savefig(filepath)