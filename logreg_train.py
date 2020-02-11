import sys
import pickle
from FileLoader import FileLoader
from LogisticRegression import LogisticRegression
import numpy as np

def setTheta(thetaG, thetaR, thetaS, thetaH):
    with open('cache', 'wb') as fichier:
        my_pickler = pickle.Pickler(fichier)
        my_pickler.dump(thetaG)
        my_pickler.dump(thetaR)
        my_pickler.dump(thetaS)
        my_pickler.dump(thetaH)
        fichier.close()

def training(data):
    data = data.dropna()#for later : should drop only NaN line in selected columns
    x_train = data[['Astronomy', 'Herbology', 'Defense Against the Dark Arts', 'Ancient Runes', 'Charms']].to_numpy()
    y = np.array(data['Hogwarts House'])

    gryffindor = LogisticRegression(alpha=0.0000001, max_iter=150, verbose=True, learning_rate='constant')
    y_train = np.where(y == 'Gryffindor', 1, 0)
    #np.set_printoptions(threshold=sys.maxsize)
    #print(y_train)
    gryffindor.fit(x_train, y_train)
    print(gryffindor.thetas)

    ravenclaw = LogisticRegression(alpha=0.0000001, max_iter=150, verbose=True, learning_rate='constant')
    y_train = np.where(y == 'Ravenclaw', 1, 0)
    ravenclaw.fit(x_train, y_train)
    print(ravenclaw.thetas)

    slytherin = LogisticRegression(alpha=0.0000001, max_iter=150, verbose=True, learning_rate='constant')
    y_train = np.where(y == 'Slytherin', 1, 0)
    slytherin.fit(x_train, y_train)
    print(slytherin.thetas)

    hufflepuff = LogisticRegression(alpha=0.0000001, max_iter=150, verbose=True, learning_rate='constant')
    y_train = np.where(y == 'Hufflepuff', 1, 0)
    hufflepuff.fit(x_train, y_train)
    print(hufflepuff.thetas)
    setTheta(gryffindor.thetas, ravenclaw.thetas, slytherin.thetas, hufflepuff.thetas)

if __name__ == "__main__":
    if len(sys.argv) == 2:
        loader = FileLoader()
        data = loader.load(str(sys.argv[1]))
        training(data)
    else:
        print("Usage : python logreg_train.py path_file")
