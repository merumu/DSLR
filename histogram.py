import sys
import numpy as np
from FileLoader import FileLoader
import matplotlib.pyplot as plt

def getHouse(data, row, house):
    house = data[data["Hogwarts House"]==house]
    return house[row].dropna()

def histogram(data):
    fig, axs = plt.subplots(3, 5, figsize=(22, 12))
    i = 0
    j = 0
    for row in data:
        if all(isinstance(x, float) for x in data[row]):
            Gryffindor = getHouse(data, row, 'Gryffindor')
            Slytherin = getHouse(data, row, 'Slytherin')
            Ravenclaw = getHouse(data, row, 'Ravenclaw')
            Hufflepuff = getHouse(data, row, 'Hufflepuff')
            if len(Gryffindor) > 0 and len(Slytherin) > 0 and len(Ravenclaw) > 0 and len(Hufflepuff) > 0:
                axs[i][j].set_title(str(row))
                axs[i][j].hist(np.hstack(Gryffindor), histtype='bar', alpha=0.3, label='Gryffindor')
                axs[i][j].hist(np.hstack(Slytherin), histtype='bar', alpha=0.3, label='Slytherin')
                axs[i][j].hist(np.hstack(Ravenclaw), histtype='bar', alpha=0.3, label='Ravenclaw')
                axs[i][j].hist(np.hstack(Hufflepuff), histtype='bar', alpha=0.3, label='Hufflepuff')
                axs[i][j].legend(prop={'size': 6})
                j += 1
                if j > 4:
                    i += 1
                    j = 0
    if i < 3 and j < 4:
        plt.delaxes(axs[2][3])
        plt.delaxes(axs[2][4])
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) == 2:
        loader = FileLoader()
        data = loader.load(str(sys.argv[1]))
        histogram(data)
    else:
        print("Usage : python histogram.py path_file")