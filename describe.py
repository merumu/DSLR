import sys
import pandas
import numpy
from FileLoader import FileLoader

def mean(data):
    total = 0
    for value in data:
        total += value
    if len(data) > 0:
        return total / len(data)
    return None

def std(data):
    n = 0
    summ = 0
    moy = mean(data)
    for value in data:
        summ += (value - moy) ** 2
        n += 1
    if n > 1:
        return numpy.sqrt(summ / (n - 1))
    return None

def mini(data):
    if len(data) > 0:
        tmp = data[data.index[0]]
        for value in data:
            if tmp > value:
                tmp = value
        return tmp
    return None

def maxi(data):
    if len(data) > 0:
        tmp = data[data.index[0]]
        for value in data:
            if tmp < value:
                tmp = value
        return tmp
    return None

def firstQuartile(data):
    tmp = data.sort_values()
    n = len(tmp)
    q1 = (n + 3) // 4
    index = 1
    for value in tmp:
        if index == q1:
            return value
        index += 1
    return None

def median(data):
    tmp = data.sort_values()
    n = len(tmp)
    q2 = (n + 1) // 2
    index = 1
    for value in tmp:
        if index == q2:
            return value
        index += 1
    return None

def thirdQuartile(data):
    tmp = data.sort_values()
    n = len(tmp)
    q3 = (3 * n + 1) // 4
    index = 1
    for value in tmp:
        if index == q3:
            return value
        index += 1
    return None

def describe(data):
    feature = pandas.DataFrame(index=['Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max'])
    n = 1
    for row in data:
        if all(isinstance(x, float) for x in data[row]):
            tmp = data[row].dropna()
            if len(tmp) > 0:
                try:
                    feature['Feature ' + str(n)] = [len(tmp), mean(tmp), std(tmp), mini(tmp), firstQuartile(tmp), median(tmp), thirdQuartile(tmp), maxi(tmp)]
                    #feature['pandas ' + str(n)] = [tmp.count(), tmp.mean(), tmp.std(), min(tmp), tmp.quantile(.25), tmp.quantile(.5), tmp.quantile(.75), max(tmp)]
                except:
                    print("Error: Overflow from a too large value in data in feature " + str(n))
                n += 1
    print(feature)

if __name__ == "__main__":
    if len(sys.argv) == 2:
        loader = FileLoader()
        data = loader.load(str(sys.argv[1]))
        describe(data)
    else:
        print("Usage : python describe.py path.csv")