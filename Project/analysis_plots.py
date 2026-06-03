import matplotlib.pyplot as plt

def plot_metrics_vs_ppc(results, ppc_values):
    accuracies = [results[p]['test_accuracy'].mean() * 100 for p in ppc_values]
    f1_scores = [results[p]['test_f1_macro'].mean() * 100 for p in ppc_values]
    fit_times = [results[p]['fit_time'].mean() for p in ppc_values]

    fig, ax = plt.subplots(figsize=(8, 6))

    color_bar = '#7fcdbb'
    color_acc = '#41b6c4'
    color_f1 = '#225ea8'

    max_time = max(fit_times) if max(fit_times) > 0 else 1
    normalized_times = [(t / max_time) * 60 for t in fit_times]

    bars = ax.bar(ppc_values, normalized_times, width=1.0, color=color_bar,
                  alpha=0.6, label='Relative Fit Time')

    for bar, time_val in zip(bars, fit_times):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 1.5,
                f'{time_val:.2f}s', ha='center', va='bottom', fontsize=9)

    ax.plot(ppc_values, accuracies, marker='o', linestyle='-', color=color_acc,
            linewidth=2, label='Accuracy')
    ax.plot(ppc_values, f1_scores, marker='s', linestyle='--', color=color_f1,
            linewidth=2, label='F1-Score')

    ax.set_xlabel('Pixels Per Cell (PPC)', fontsize=12)
    ax.set_ylabel('Score (%)', fontsize=12)
    ax.set_xticks(ppc_values)
    ax.set_xticklabels([f'{ppc}×{ppc}' for ppc in ppc_values])
    ax.set_ylim(0, 105)
    ax.grid(True, linestyle='--', alpha=0.4)

    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25),
              ncol=3, frameon=False, fontsize=11)

    plt.title('Classification Metrics and Fit Time by PPC Configuration', fontsize=14, pad=15)
    plt.tight_layout()
    plt.show()