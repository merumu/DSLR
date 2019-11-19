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
    n = 0
    for value in data:
        n += value
    return n

def mean(data):



def describe(data):
    feature = pandas.DataFrame(index=['Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max'])
    n = 1
    for row in data:
        if all(isinstance(x, float) for x in data[row]):
            tmp = data.dropna()
            feature['Feature ' + str(n)] = [count(tmp), mean(tmp), 3, 4, 5, 6, 7, 8]
            n += 1
    print(feature)

if __name__ == "__main__":
    if len(sys.argv) == 2:
        loader = FileLoader()
        data = loader.load(str(sys.argv[1]))
        describe(data)
    else:
        print("Usage : python describe.py path_file")