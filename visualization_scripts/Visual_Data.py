# pandas
import pandas as pd
# numpy
import numpy
# matplotlib
import matplotlib
# Load Libraries 
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot

B_data = pd.read_csv("c_train.csv")

# confirm shape
print(B_data.shape)

# descriptions (quick summary)
print(B_data.describe())

# promoted hard count vs not
print(B_data.groupby('is_promoted').size())

# box and whisker plots to identify input data distribution
B_data.plot(kind='box', subplots=True, layout=(3,4), sharex=False, sharey=False)
pyplot.show()

# histograms to help us identify distibutions to exploit in algorithm
B_data.hist(layout=(3,4))
pyplot.show()

# scatter plot matrix
scatter_matrix(B_data)
pyplot.show()