import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict, GroupKFold
from sklearn.ensemble import RandomForestClassifier

# ----- FONT SETTINGS -----
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 11,
    'axes.titlesize': 14
})

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def plot_confusion_matrix(X, y, groups, emotions_map, ppc, cv_folds=5, random_state=7):
    class_names = list(emotions_map.keys())
    model = RandomForestClassifier(n_estimators=100, n_jobs=5, random_state=random_state)
    y_pred = cross_val_predict(model, X, y, groups=groups, cv=GroupKFold(n_splits=cv_folds))

    cm = confusion_matrix(y, y_pred, normalize='true')

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Prediction Proportion'})

    plt.ylabel("True Emotion", fontweight='bold', labelpad=25)
    plt.xlabel("Predicted Emotion", fontweight='bold', labelpad=25)

    plt.tight_layout()

    output_dir = os.path.join(BASE_DIR, "data")
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"confusion.pdf")
    plt.savefig(save_path, format='pdf', bbox_inches='tight')

    plt.show()
