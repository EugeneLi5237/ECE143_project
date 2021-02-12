# Import required library
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Import the CSV file into Python
A_data = pd.read_csv("c_train.csv")
A_data = A_data.dropna()
print(A_data.shape)

# Run correlation on the data and plot heatmap
heatmap = sns.heatmap(A_data.corr(), cmap="RdYlGn", cbar_kws = dict(use_gridspec=False, location="top", label='Correlation'), xticklabels=2, annot=False, linewidths=.5)
plt.xticks(rotation=0) 
plt.show()
