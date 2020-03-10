import sys
import numpy as np
sys.path.append('../utils')
from FileLoader import FileLoader
import matplotlib.pyplot as plt
import pandas as pd
#from pandas.plotting import scatter_matrix
import seaborn as sns

def pairPlot(data):
    try:
        df = data.drop(columns=['Index','First Name','Last Name','Birthday','Best Hand'])
    except:
        print("Error: wrong column name in data")
        exit()
    #pd.plotting.scatter_matrix(df, alpha = 0.2, figsize = (13, 13), diagonal = 'hist')
    sns.set(style="ticks")
    sns.set_context("notebook", font_scale=1.2)
    g = sns.PairGrid(df, hue="Hogwarts House", height=1)
    g.map_diag(plt.hist, alpha=0.8)
    g.map_offdiag(plt.scatter, s=1)
    g.add_legend()
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) == 2:
        loader = FileLoader()
        data = loader.load(str(sys.argv[1]))
        pairPlot(data)
    else:
        print("Usage : python pair_plot.py path.csv")