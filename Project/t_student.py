import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# ----- FONT SETTINGS -----
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 11,
    'axes.titlesize': 14
})

f1_standard = [70.76, 71.60, 61.86, 62.75, 57.00]
f1_balanced = [73.01, 72.54, 58.42, 64.85, 59.16]

data = {
    'F1-Score (%)': f1_standard + f1_balanced,
    'Model': ['Standard'] * 5 + ['Balanced'] * 5
}
df = pd.DataFrame(data)

plt.figure(figsize=(8, 6))

ax = sns.boxplot(x='Model', y='F1-Score (%)', data=df,
                 width=0.5, palette="YlGnBu", showmeans=True,
                 meanprops={"marker":"^", "markerfacecolor":"white",
                            "markeredgecolor":"black", "markersize":"10"},
                 hue='Model', legend=False)

sns.stripplot(x='Model', y='F1-Score (%)', data=df,
              color="white", edgecolor="black", linewidth=1, size=8, jitter=False)

ax.set_axisbelow(True)
ax.grid(axis='y', color='lightgray', linestyle='-')

plt.ylabel('F1-Score (%)')
plt.xlabel('Class Weight Configuration')
plt.ylim(50, 80)

plt.tight_layout()

output_dir = "Figures"
os.makedirs(output_dir, exist_ok=True)
save_path = os.path.join(output_dir, "t_student.pdf")
plt.savefig(save_path, format='pdf', bbox_inches='tight')

plt.show()