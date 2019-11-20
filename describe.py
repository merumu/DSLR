import sys
import pandas
import numpy

class FileLoader:
    def __init__(self):
        pass

    def load(self, path):
        csv_file = pandas.read_csv(path)
        df = pandas.DataFrame(csv_file)
        print("Loading dataset of dimensions {} x {}".format(len(df.axes[0]), len(df.axes[1])))
        return df

def count(data):
    total = 0
    for value in data:
        total += value
    return total

def mean(data):
    total = count(data)
    return total / len(data)

def std(data):
    n = 0
    summ = 0
    moy = mean(data)
    for value in data:
        summ += (value - moy) ** 2
        n += 1
    return (summ / n) ** 1/2

def mini(data):
    tmp = data[0]
    for value in data:
        if tmp > value:
            tmp = value
    return tmp

def maxi(data):
    tmp = data[0]
    for value in data:
        if tmp < value:
            tmp = value
    return tmp

def firstQuartile(data):
    #tmp = data.sort_values()
    #n = len(tmp)
    #print((n + 3) // 4)
    #print(tmp[(n + 3) // 4])
    return 0

def median():
    pass

def thirdQuartile():
    pass

def describe(data):
    feature = pandas.DataFrame(index=['Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max'])
    n = 1
    for row in data:
        if all(isinstance(x, float) for x in data[row]):
            tmp = data[row].dropna()
            if len(tmp) > 0:
                feature['Feature ' + str(n)] = [count(tmp), mean(tmp), std(tmp), mini(tmp), firstQuartile(tmp), 6, 7, maxi(tmp)]
                n += 1
    print(feature)

if __name__ == "__main__":
    if len(sys.argv) == 2:
        loader = FileLoader()
        data = loader.load(str(sys.argv[1]))
        describe(data)
    else:
        print("Usage : python describe.py path_file")