import sys
import numpy as np
from FileLoader import FileLoader
import matplotlib.pyplot as plt

def histogram(data):
    for row in data:
        if all(isinstance(x, float) for x in data[row]):
            tmp = data[row].dropna()
            if len(tmp) > 0:
                a = np.hstack(tmp)
                _ = plt.hist(a, bins='auto')
                plt.title(str(row))
                plt.show()

if __name__ == "__main__":
    if len(sys.argv) == 2:
        loader = FileLoader()
        data = loader.load(str(sys.argv[1]))
        histogram(data)
    else:
        print("Usage : python histogram.py path_file")