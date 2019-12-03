import sys
import pickle
from FileLoader import FileLoader

def getTheta():
    try:
        with open('cache', 'rb') as fichier:
            my_unpickler = pickle.Unpickler(fichier)
            theta0 = my_unpickler.load()
            theta1 = my_unpickler.load()
            fichier.close()
    except:
        theta0 = 0
        theta1 = 0
    return (theta0, theta1)

def setTheta(theta0, theta1):
    with open('cache', 'wb') as fichier:
        my_pickler = pickle.Pickler(fichier)
        my_pickler.dump(theta0)
        my_pickler.dump(theta1)
        fichier.close()

def training(data):
    pass

if __name__ == "__main__":
    theta0, theta1 = getTheta()
    if theta0 == 0 and theta1 == 0:
        setTheta(theta0, theta1)
    if len(sys.argv) == 2:
        loader = FileLoader()
        data = loader.load(str(sys.argv[1]))
        training(data)
    else:
        print("Usage : python logreg_train.py path_file")
