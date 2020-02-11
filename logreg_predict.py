import sys
import pickle
from FileLoader import FileLoader
import numpy as np
from logreg_train import setTheta

def getTheta(size):
    try:
        with open('cache', 'rb') as fichier:
            my_unpickler = pickle.Unpickler(fichier)
            thetaG = my_unpickler.load()
            thetaR = my_unpickler.load()
            thetaS = my_unpickler.load()
            thetaH = my_unpickler.load()
            fichier.close()
    except:
        thetaG = np.zeros(size)
        thetaR = np.zeros(size)
        thetaS = np.zeros(size)
        thetaH = np.zeros(size)
    return (thetaG, thetaR, thetaS, thetaH)

def delete_nan(stud, thetaG, thetaR, thetaS, thetaH):
    delete = 0
    for n in range(5):
        if n + delete < 5 and np.isnan(stud[n]):#delete nan in stud and corresponding thetas
            stud = np.delete(stud, n)
            thetaG = np.delete(thetaG, n + 1)
            thetaR = np.delete(thetaR, n + 1)
            thetaS = np.delete(thetaS, n + 1)
            thetaH = np.delete(thetaH, n + 1)
            n -= 1
            delete += 1
    stud = stud.reshape((1,5 - delete))
    return (stud, thetaG, thetaR, thetaS, thetaH)

def predict(data):
    x_test = data[['Astronomy', 'Herbology', 'Defense Against the Dark Arts', 'Ancient Runes', 'Charms']].to_numpy()
    for stud in x_test:
        (thetaG, thetaR, thetaS, thetaH) = getTheta(stud.shape[0] + 1)
        (stud, thetaG, thetaR, thetaS, thetaH) = delete_nan(stud, thetaG, thetaR, thetaS, thetaH)
        stud = np.insert(stud, 0, 1, axis=1)
        g_score = stud.dot(thetaG)
        r_score = stud.dot(thetaR)
        s_score = stud.dot(thetaS)
        h_score = stud.dot(thetaH)
        print("score :")
        print(g_score)
        print(r_score)
        print(s_score)
        print(h_score)

if __name__ == "__main__":
    if len(sys.argv) == 2:
        loader = FileLoader()
        data = loader.load(str(sys.argv[1]))
        predict(data)
    else:
        print("Usage : python logreg_predict.py path_file")