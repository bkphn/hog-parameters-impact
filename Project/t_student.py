import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

f1_standard = [70.76, 71.60, 61.86, 62.75, 57.00]
f1_balanced = [73.01, 72.54, 58.42, 64.85, 59.16]

data = {
    'F1-Score (%)': f1_standard + f1_balanced,
    'Model': ['Standard'] * 5 + ['Balanced'] * 5
}
df = pd.DataFrame(data)

sns.set_theme(style="whitegrid")
plt.figure(figsize=(8, 6))

ax = sns.boxplot(x='Model', y='F1-Score (%)', data=df,
                 width=0.5, palette="Set2", showmeans=True,
                 meanprops={"marker":"^", "markerfacecolor":"white",
                            "markeredgecolor":"black", "markersize":"10"})

sns.swarmplot(x='Model', y='F1-Score (%)', data=df, color=".25", size=8)

plt.title('F1-Score Distribution in 5-Fold Cross-Validation (PPC=8)', fontsize=14, pad=15)
plt.ylabel('F1-Score (%)', fontsize=12)
plt.xlabel('Class Weight Configuration', fontsize=12)
plt.ylim(50, 80)

plt.text(0.5, 78, 'p-value = 0.50 (No statistical significance)',
         horizontalalignment='center', fontsize=11,
         bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))

plt.tight_layout()
plt.show()