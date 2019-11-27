import sys
import numpy as np
from FileLoader import FileLoader
import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import scatter_matrix

def pairPlot(data):
    df = data.drop(columns=['Index','Hogwarts House','First Name','Last Name','Birthday','Best Hand'])
    pd.plotting.scatter_matrix(df, alpha = 0.2, figsize = (13, 13), diagonal = 'hist')
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) == 2:
        loader = FileLoader()
        data = loader.load(str(sys.argv[1]))
        pairPlot(data)
    else:
        print("Usage : python histogram.py path_file")