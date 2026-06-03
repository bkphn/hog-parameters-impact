import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import RandomForestClassifier


def plot_confusion_matrix(X, y, emotions_map, ppc, cv_folds=5, random_state=7):
    class_names = list(emotions_map.keys())
    model = RandomForestClassifier(n_estimators=100, n_jobs=5, random_state=random_state)
    y_pred = cross_val_predict(model, X, y, cv=cv_folds)

    cm = confusion_matrix(y, y_pred, normalize='true')

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Prediction Proportion'})

    plt.title(f"Normalized Confusion Matrix (Random Forest, PPC={ppc}×{ppc})", fontsize=14, pad=15)
    plt.ylabel("True Emotion (True Label)", fontsize=12, fontweight='bold')
    plt.xlabel("Predicted Emotion (Predicted Label)", fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.show()
