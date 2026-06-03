import os
import pickle
import numpy as np

from preprocessing import preprocess_data
from random_forest import train
from matrices import plot_confusion_matrix

# ----- MODEL SETTINGS -----
PREPROCESSING = True
TRAIN_MODEL = False
VALIDATION_CURVE = False

# ----- MATRICES SETTINGS -----
GENERATE_MATRICES = True
ONLY_FOR_8x8 = True

# ----- CMD COLOR SETTINGS -----
RED = '\033[91m \x1B[3m'
YELLOW = '\033[93m \x1B[3m'
GREEN = '\033[92m \x1B[3m'
RESET = '\033[0m \x1B[0m'

# ----- PATHS TO DATA -----
DATA_PATH = "Data"

# ------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    emotions_map = {"anger": 0, "contempt": 1, "disgust": 2,
                    "fear": 3, "happy": 4, "sadness": 5, "surprise": 6}

    PPC = [2, 4, 6, 8, 10, 12, 14, 16]
    results = {}

    for ppc in PPC:
        print(f"{GREEN}Rozpoczynam ekstrakcję cech HOG dla {ppc}x{ppc}...\n{RESET}")
        features, labels = preprocess_data(emotions_map, DATA_PATH, ppc)
        X = np.array(features, dtype=np.float32)
        y = np.array(labels, dtype=np.float32)
        print(f"{GREEN}Ekstrakcja cech zakończona pomyślnie.{RESET}")

        print(f"{GREEN}Rozpoczynam klasyfikację lasem losowym dla {ppc}x{ppc}...{RESET}")
        metrics = train(X, y)
        results[ppc] = metrics
        print(f"{GREEN}Zakończono klasyfikację.{RESET}")

        print(f"\n--- REZULTATY DLA {ppc}x{ppc} ---")
        print(f"Accuracy:  {metrics['test_accuracy'].mean() * 100:.2f}%")
        print(f"Precision: {metrics['test_precision_macro'].mean() * 100:.2f}%")
        print(f"Recall:    {metrics['test_recall_macro'].mean() * 100:.2f}%")
        print(f"F1:        {metrics['test_f1_macro'].mean() * 100:.2f}%")
        print(f"Czas:      {metrics['fit_time'].mean():.4f} s")

        if GENERATE_MATRICES and not ONLY_FOR_8x8:
            print(f"{GREEN}Generuję macierze pomyłek {ppc}x{ppc}...{RESET}")
            plot_confusion_matrix(X, y, emotions_map, ppc)
            print(f"{GREEN}Macierz pomyłek wygenerowanie pomyślnie.{RESET}")

    if GENERATE_MATRICES and ONLY_FOR_8x8:
        print(f"{GREEN}Generuję macierz pomyłek {ppc}x{ppc}...{RESET}")
        plot_confusion_matrix(X, y, emotions_map, 8)
        print(f"{GREEN}Macierz pomyłek wygenerowanie pomyślnie.{RESET}")