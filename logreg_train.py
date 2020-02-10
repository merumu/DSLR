import sys
import pickle
from FileLoader import FileLoader
from LogisticRegression import LogisticRegression
import numpy as np

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
    x_train = data[['Charms' ,'Flying']].to_numpy()
    y = np.array(data['Hogwarts House'])

    gryffindor = LogisticRegression(alpha=0.00001, max_iter=150, verbose=True, learning_rate='constant')
    y_train = np.where(y == 'Gryffindor', 1, 0)
    #np.set_printoptions(threshold=sys.maxsize)
    #print(y_train)
    gryffindor.fit(x_train, y_train)

    ravenclaw = LogisticRegression(alpha=0.00001, max_iter=150, verbose=True, learning_rate='constant')
    y_train = np.where(y == 'Ravenclaw', 1, 0)
    ravenclaw.fit(x_train, y_train)

    slytherin = LogisticRegression(alpha=0.00001, max_iter=150, verbose=True, learning_rate='constant')
    y_train = np.where(y == 'Slytherin', 1, 0)
    slytherin.fit(x_train, y_train)

    hufflepuff = LogisticRegression(alpha=0.00001, max_iter=150, verbose=True, learning_rate='constant')
    y_train = np.where(y == 'Hufflepuff', 1, 0)
    hufflepuff.fit(x_train, y_train)
    #test
    test = np.array([-255.47823,167.38])
    test = test.reshape((1,2))
    print(gryffindor.predict(test))
    print(ravenclaw.predict(test))
    print(slytherin.predict(test))
    print(hufflepuff.predict(test))

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
