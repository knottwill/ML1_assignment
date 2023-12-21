import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, ConfusionMatrixDisplay
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

def random_forest_imputation(X, rf_params):
    X_imputed = X.copy()
    for col in X.columns:
        if X_imputed[col].isna().sum() > 0:
            # Split data into sets with and without missing values
            X_with_value = X_imputed[X_imputed[col].notna()]
            X_missing = X_imputed[X_imputed[col].isna()]

            # Separate predictors and target
            y_train = X_with_value[col]
            X_train = X_with_value.drop(col, axis=1)
            X_test = X_missing.drop(col, axis=1)

            # Fill missing values in predictors with column median
            # (median over mean since most features are not normally distributed)
            X_train = X_train.fillna(X_train.median())
            X_test = X_test.fillna(X_train.median())

            model = RandomForestRegressor(**rf_params)

            # Fit model and predict missing values
            model.fit(X_train, y_train)
            X_imputed.loc[X_missing.index, col] = model.predict(X_test)

    return X_imputed

# Function to convert classification report to DataFrame without accuracy row
def parse_classification_report(report):
    lines = report.split('\n')[2:-4]  # Exclude the last line (accuracy)
    data = [line.split() for line in lines if line]
    headers = ["Class", "Precision", "Recall", "F1-Score", "Support"]
    return pd.DataFrame(data, columns=headers)


def classifier_evaluation_plot(y_true, y_pred, classes, filepath=None):

    # calculate classification report, confusion matrix and accuracy
    class_report = classification_report(y_true, y_pred)
    print("Classification Report\n", class_report)
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