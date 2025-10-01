import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_label_distribution(labels, title="Phân bố nhãn"):
    label_counts = pd.Series(labels).value_counts()
    plt.figure(figsize=(6, 4))
    sns.barplot(x=label_counts.index, y=label_counts.values, palette="viridis")
    plt.title(title)
    plt.show()