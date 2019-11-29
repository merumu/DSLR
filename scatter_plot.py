import sys
import numpy as np
from FileLoader import FileLoader
import matplotlib.pyplot as plt
from describe import *

def getHouse(data, house):
    house = data[data["Hogwarts House"]==house]
    return house

def compare(tmp1, tmp2):
    a = mean(tmp1)
    b = mean(tmp2)
    c = abs(a - b)
    a = std(tmp1)
    b = std(tmp2)
    c += abs(a - b)
    a = mini(tmp1)
    b = mini(tmp2)
    c += abs(a - b)
    a = maxi(tmp1)
    b = maxi(tmp2)
    c += abs(a - b)
    a = firstQuartile(tmp1)
    b = firstQuartile(tmp2)
    c += abs(a - b)
    a = median(tmp1)
    b = median(tmp2)
    c += abs(a - b)
    a = thirdQuartile(tmp1)
    b = thirdQuartile(tmp2)
    c += abs(a - b)
    return c

def scatterPlot(data):
    best = None
    for row1 in data:
        if all(isinstance(x, float) for x in data[row1]):
            tmp1 = data[row1].dropna()
            if len(tmp1) > 0:
                for row2 in data:
                    if row1 != row2 and all(isinstance(x, float) for x in data[row2]):
                        tmp2 = data[row2].dropna()
                        if len(tmp2) > 0:
                            score = compare(tmp1, tmp2)
                            if best == None or score < best:
                                best = score
                                feat1 = row1
                                feat2 = row2
    df = data[['Hogwarts House',feat1,feat2]].dropna()
    #df1 = df[feat1]
    #df2 = df[feat2]
    Gryffindor = getHouse(data, 'Gryffindor')
    Slytherin = getHouse(data, 'Slytherin')
    Ravenclaw = getHouse(data, 'Ravenclaw')
    Hufflepuff = getHouse(data, 'Hufflepuff')
    plt.title(str(feat1) + " / " +str(feat2))
    #plt.scatter(df1.index, df1, alpha=0.5, label=str(feat1))
    #plt.scatter(df2.index, df2, alpha=0.5, label=str(feat2))
    plt.scatter(Gryffindor[feat1], Gryffindor[feat2], alpha=0.5, label='Gryffindor', color='r')
    plt.scatter(Slytherin[feat1], Slytherin[feat2], alpha=0.5, label='Slytherin', color='g')
    plt.scatter(Ravenclaw[feat1], Ravenclaw[feat2], alpha=0.5, label='Ravenclaw', color='b')
    plt.scatter(Hufflepuff[feat1], Hufflepuff[feat2], alpha=0.5, label='Hufflepuff', color='yellow')
    plt.xlabel(str(feat1))
    plt.ylabel(str(feat2))
    plt.legend(prop={'size': 6})
    print(str(feat1) + " and " + str(feat2) + " are the most similar features")
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) == 2:
        loader = FileLoader()
        data = loader.load(str(sys.argv[1]))
        scatterPlot(data)
    else:
        print("Usage : python histogram.py path_file")