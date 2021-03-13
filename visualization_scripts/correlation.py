import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def correlation_plot(fname):
    """
    Method to plot the correlation data from a dataset in a .csv file
    """

    assert isinstance(fname, str)
    assert len(fname) > 0
    assert fname[-4:] == ".csv"

    # Import the CSV file into Python
    A_data = pd.read_csv(fname)
    A_data = A_data.dropna()

    # Run correlation on the data and plot heatmap
    heatmap = sns.heatmap(A_data.corr(), cmap="RdYlGn", cbar_kws=dict(
        use_gridspec=False, location="top", label='Correlation'), xticklabels=2, annot=False, linewidths=.5)
    plt.xticks(rotation=0)
    plt.show()
