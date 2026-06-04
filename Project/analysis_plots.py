import os
import matplotlib.pyplot as plt

# ----- FONT SETTINGS -----
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 11,
    'axes.titlesize': 14
})


def plot_metrics_vs_ppc(results, results_svc, ppc_values):
    accuracies = [results[p]['test_accuracy'].mean() * 100 for p in ppc_values]
    f1_scores = [results[p]['test_f1_macro'].mean() * 100 for p in ppc_values]
    fit_times = [results[p]['fit_time'].mean() for p in ppc_values]

    accuracies_svc = [results_svc[p]['test_accuracy'].mean() * 100 for p in ppc_values]
    f1_scores_svc = [results_svc[p]['test_f1_macro'].mean() * 100 for p in ppc_values]
    fit_times_svc = [results_svc[p]['fit_time'].mean() for p in ppc_values]

    fig, ax = plt.subplots(figsize=(8, 6))

    color_bar = '#7fcdbb'
    color_acc = '#41b6c4'
    color_f1 = '#225ea8'

    color_acc_svc = '#be493b'
    color_f1_svc = '#dda157'
    color_bar_svc = '#803244'

    max_time = max(fit_times) if max(fit_times) > 0 else 1
    normalized_times = [(t / max_time) * 60 for t in fit_times]

    x_pos = range(len(ppc_values))

    bars = ax.bar(x_pos, normalized_times, width=0.6, color=color_bar,
                  alpha=0.6, label='RF Relative Fit Time')

    for bar, time_val in zip(bars, fit_times):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 1.5,
                f'{time_val:.2f}s', ha='center', va='bottom')

    ax.plot(x_pos, accuracies, marker='o', linestyle='-', color=color_acc,
            linewidth=2, label='RF Accuracy')
    ax.plot(x_pos, f1_scores, marker='s', linestyle='--', color=color_f1,
            linewidth=2, label='RF F1-Score')

    ax.plot(x_pos, accuracies_svc, marker='o', linestyle='-', color=color_acc_svc,
            linewidth=2, label='SVC Accuracy')
    ax.plot(x_pos, f1_scores_svc, marker='s', linestyle='--', color=color_f1_svc,
            linewidth=2, label='SVC F1-Score')

    ax.set_xlabel('Pixels Per Cell (PPC)')
    ax.set_ylabel('Score (%)')

    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'{ppc}×{ppc}' for ppc in ppc_values])

    ax.set_ylim(0, 105)
    ax.grid(True, linestyle='--', alpha=0.4)

    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25),
              ncol=3, frameon=False)

    plt.tight_layout()

    output_dir = "Figures"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "chart.pdf")
    plt.savefig(save_path, format='pdf', bbox_inches='tight')

    plt.show()
