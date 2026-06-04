import os
import pickle
import numpy as np

from preprocessing import preprocess_data
from random_forest import train
from matrices import plot_confusion_matrix
from analysis_plots import plot_metrics_vs_ppc
from t_student import perform_statistical_analysis, plot_svc_boxplot, plot_rf_boxplot
from svc import train_svc

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

    PPC = [2, 4, 6, 8, 12, 16]
    results_rf = {}
    results_svc = {}

    for ppc in PPC:
        print(f"{GREEN}Rozpoczynam ekstrakcję cech HOG dla {ppc}x{ppc}...\n{RESET}")
        features, labels = preprocess_data(emotions_map, DATA_PATH, ppc)
        X = np.array(features, dtype=np.float32)
        y = np.array(labels, dtype=np.float32)
        print(f"{GREEN}Ekstrakcja cech zakończona pomyślnie.{RESET}")

        print(f"{GREEN}Rozpoczynam klasyfikację lasem losowym dla {ppc}x{ppc}...{RESET}")
        metrics_rf = train(X, y)
        results_rf[ppc] = metrics_rf
        print(f"{GREEN}Zakończono klasyfikację lasem losowym.{RESET}")

        print(f"{GREEN}Rozpoczynam klasyfikację Support Vector Classification dla {ppc}x{ppc}...{RESET}")
        metrics_svc = train_svc(X, y)
        results_svc[ppc] = metrics_svc
        print(f"{GREEN}Zakończono klasyfikację SVC.{RESET}")

        print(f"\n--- REZULTATY DLA RF {ppc}x{ppc} ---")
        print(f"Accuracy:  {metrics_rf['test_accuracy'].mean() * 100:.2f}%")
        print(f"Precision: {metrics_rf['test_precision_macro'].mean() * 100:.2f}%")
        print(f"Recall:    {metrics_rf['test_recall_macro'].mean() * 100:.2f}%")
        print(f"F1:        {metrics_rf['test_f1_macro'].mean() * 100:.2f}%")
        print(f"Czas:      {metrics_rf['fit_time'].mean():.4f} s")

        print(f"\n--- REZULTATY DLA SVC {ppc}x{ppc} ---")
        print(f"Accuracy:  {metrics_svc['test_accuracy'].mean() * 100:.2f}%")
        print(f"Precision: {metrics_svc['test_precision_macro'].mean() * 100:.2f}%")
        print(f"Recall:    {metrics_svc['test_recall_macro'].mean() * 100:.2f}%")
        print(f"F1:        {metrics_svc['test_f1_macro'].mean() * 100:.2f}%")
        print(f"Czas:      {metrics_svc['fit_time'].mean():.4f} s")

        if GENERATE_MATRICES and not ONLY_FOR_8x8:
            print(f"{GREEN}Generuję macierze pomyłek {ppc}x{ppc}...{RESET}")
            plot_confusion_matrix(X, y, emotions_map, ppc)
            print(f"{GREEN}Macierz pomyłek wygenerowanie pomyślnie.{RESET}")

        if GENERATE_MATRICES and ONLY_FOR_8x8 and ppc == 8:
            print(f"{GREEN}Generuję macierz pomyłek {ppc}x{ppc}...{RESET}")
            plot_confusion_matrix(X, y, emotions_map, 8)
            print(f"{GREEN}Macierz pomyłek wygenerowanie pomyślnie.{RESET}")

    print(f"{GREEN}Generuję wykres zbiorczy analizy wpływu PPC...{RESET}")
    plot_metrics_vs_ppc(results_rf, results_svc, PPC)

    print(f"{GREEN}Przeprowadzam test t-studenta...{RESET}")
    perform_statistical_analysis(results_rf, results_svc)
    plot_rf_boxplot(results_rf)
    plot_svc_boxplot(results_svc)
    print(f"Test przeprowadzono pomyślnie.{RESET}")