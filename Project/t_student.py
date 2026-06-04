import os
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
                 width=0.5, palette="YlGnBu", showmeans=True,
                 meanprops={"marker":"^", "markerfacecolor":"white",
                            "markeredgecolor":"black", "markersize":"10"},
                 hue='Model', legend=False)

sns.stripplot(x='Model', y='F1-Score (%)', data=df,
              color="white", edgecolor="black", linewidth=1, size=8, jitter=False)

plt.ylabel('F1-Score (%)', fontsize=12)
plt.xlabel('Class Weight Configuration', fontsize=12)
plt.ylim(50, 80)

plt.tight_layout()

output_dir = "Figures"
os.makedirs(output_dir, exist_ok=True)
save_path = os.path.join(output_dir, f"t_student.pdf")
plt.savefig(save_path, format='pdf', bbox_inches='tight')

plt.show()