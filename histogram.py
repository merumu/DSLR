import sys
from FileLoader import FileLoader

def histogram(data):
    for row in data:
        if all(isinstance(x, float) for x in data[row]):
            print("draw histogram of " + str(row))

if __name__ == "__main__":
    if len(sys.argv) == 2:
        loader = FileLoader()
        data = loader.load(str(sys.argv[1]))
        histogram(data)
    else:
        print("Usage : python histogram.py path_file")