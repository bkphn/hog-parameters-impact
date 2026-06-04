from scipy import stats
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 11,
    'axes.titlesize': 14
})

def perform_statistical_analysis(results_rf, results_svc):
    ppc_best = 8
    ppc_worst = 16

    f1_rf_best = results_rf[ppc_best]['test_f1_macro']
    f1_rf_worst = results_rf[ppc_worst]['test_f1_macro']

    t_stat_rf, p_val_rf = stats.ttest_rel(f1_rf_best, f1_rf_worst)

    print(f"\n1. Wpływ PPC na Random Forest ({ppc_best}x{ppc_best} vs {ppc_worst}x{ppc_worst})")
    print(f"p-value: {p_val_rf:.4f}")
    if p_val_rf < 0.05:
        print("Wynik istotny statystycznie.")
    else:
        print("Wynik nieistotny statystycznie.")

    f1_svc_best = results_svc[ppc_best]['test_f1_macro']
    f1_svc_worst = results_svc[ppc_worst]['test_f1_macro']

    t_stat_svc, p_val_svc = stats.ttest_rel(f1_svc_best, f1_svc_worst)

    print(f"\n2. Wpływ PPC na SVC ({ppc_best}x{ppc_best} vs {ppc_worst}x{ppc_worst})")
    print(f"p-value: {p_val_svc:.4f}")
    if p_val_svc < 0.05:
        print("Wynik istotny statystycznie.")
    else:
        print("Wynik nieistotny statystycznie.")


def plot_rf_boxplot(results_rf):
    f1_rf_8 = [val * 100 for val in results_rf[8]['test_f1_macro']]
    f1_rf_16 = [val * 100 for val in results_rf[16]['test_f1_macro']]

    plt.figure(figsize=(6, 6))
    df_rf = pd.DataFrame({'F1-Score (%)': f1_rf_8 + f1_rf_16, 'Konfiguracja': ['RF (8x8)'] * 5 + ['RF (16x16)'] * 5})

    sns.boxplot(x='Konfiguracja', y='F1-Score (%)', data=df_rf, width=0.4, palette="YlGnBu", showmeans=True,
                meanprops={"marker": "^", "markerfacecolor": "white", "markeredgecolor": "black", "markersize": "10"},
                hue='Konfiguracja', legend=False)
    sns.stripplot(x='Konfiguracja', y='F1-Score (%)', data=df_rf, color="white", edgecolor="black", linewidth=1, size=8,
                  jitter=False)

    plt.ylabel('F1-Score (%)', fontweight='bold')
    plt.xlabel('')
    plt.grid(axis='y', color='lightgray', linestyle='-')
    plt.ylim(min(df_rf['F1-Score (%)']) - 2, max(df_rf['F1-Score (%)']) + 2)

    plt.tight_layout()
    os.makedirs("Figures", exist_ok=True)
    plt.savefig("Figures/test_rf.pdf", format='pdf', bbox_inches='tight')
    plt.show()


def plot_svc_boxplot(results_svc):
    f1_svc_8 = [val * 100 for val in results_svc[8]['test_f1_macro']]
    f1_svc_16 = [val * 100 for val in results_svc[16]['test_f1_macro']]

    plt.figure(figsize=(6, 6))
    df_svc = pd.DataFrame(
        {'F1-Score (%)': f1_svc_8 + f1_svc_16, 'Konfiguracja': ['SVC (8x8)'] * 5 + ['SVC (16x16)'] * 5})

    sns.boxplot(x='Konfiguracja', y='F1-Score (%)', data=df_svc, width=0.4, palette="OrRd", showmeans=True,
                meanprops={"marker": "^", "markerfacecolor": "white", "markeredgecolor": "black", "markersize": "10"},
                hue='Konfiguracja', legend=False)
    sns.stripplot(x='Konfiguracja', y='F1-Score (%)', data=df_svc, color="white", edgecolor="black", linewidth=1,
                  size=8, jitter=False)

    plt.ylabel('F1-Score (%)', fontweight='bold')
    plt.xlabel('')
    plt.grid(axis='y', color='lightgray', linestyle='-')
    plt.ylim(min(df_svc['F1-Score (%)']) - 2, max(df_svc['F1-Score (%)']) + 2)

    plt.tight_layout()
    os.makedirs("Figures", exist_ok=True)
    plt.savefig("Figures/test_svc.pdf", format='pdf', bbox_inches='tight')
    plt.show()