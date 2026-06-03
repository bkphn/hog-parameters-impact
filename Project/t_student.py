##TODO kod od gemini, sprawdzić, zweryfikować i przeredagować

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 1. Przygotowanie danych z Twojego eksperymentu
f1_standard = [70.76, 71.60, 61.86, 62.75, 57.00]
f1_balanced = [73.01, 72.54, 58.42, 64.85, 59.16]

# 2. Przekształcenie do formatu DataFrame (wymagane przez Seaborn)
data = {
    'F1-Score (%)': f1_standard + f1_balanced,
    'Model': ['Standardowy'] * 5 + ['Zbalansowany'] * 5
}
df = pd.DataFrame(data)

# 3. Konfiguracja stylu wykresu (naukowy i czytelny)
sns.set_theme(style="whitegrid")
plt.figure(figsize=(8, 6))

# 4. Generowanie Boxplotu
# Parametr showmeans=True dodaje znacznik średniej (często wymagane w publikacjach)
ax = sns.boxplot(x='Model', y='F1-Score (%)', data=df,
                 width=0.5, palette="Set2", showmeans=True,
                 meanprops={"marker":"^", "markerfacecolor":"white",
                            "markeredgecolor":"black", "markersize":"10"})

# 5. Nałożenie punktów (Swarmplot) by pokazać konkretne wartości z 5 foldów
sns.swarmplot(x='Model', y='F1-Score (%)', data=df, color=".25", size=8)

# 6. Opisy i estetyka
plt.title('Rozkład metryki F1-Score w 5-krotnej walidacji krzyżowej (PPC=8)', fontsize=14, pad=15)
plt.ylabel('F1-Score (%)', fontsize=12)
plt.xlabel('Konfiguracja wag klasyfikatora', fontsize=12)
plt.ylim(50, 80) # Ustawienie sztywnych ram osi Y ułatwia analizę i nie fałszuje różnic

# Opcjonalne dodanie informacji o p-value bezpośrednio na wykresie
plt.text(0.5, 78, 'p-value = 0.50 (Brak istotności statystycznej)',
         horizontalalignment='center', fontsize=11,
         bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))

plt.tight_layout()
plt.show()