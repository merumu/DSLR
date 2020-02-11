import sys
import pickle
from FileLoader import FileLoader
import numpy as np
from logreg_train import setTheta, normalize
import csv

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

def delete_nan(size, stud):
    for n in range(size):
        if np.isnan(stud[n]):
            stud[n] = 0
    stud = stud.reshape((1, size))
    return stud

def predict(data):
    #x_data = data[['Astronomy','Herbology','Defense Against the Dark Arts','Divination','Muggle Studies','Ancient Runes','History of Magic','Transfiguration','Care of Magical Creatures','Charms']]
    x_data = data[['Arithmancy','Astronomy','Herbology','Defense Against the Dark Arts','Divination','Muggle Studies','Ancient Runes','History of Magic','Transfiguration','Potions','Care of Magical Creatures','Charms','Flying']]
    x_norm = normalize(x_data)
    x_test = x_norm.to_numpy()
    with open('houses.csv', 'w', newline='') as csvfile:
        fieldnames = ['Index', 'Hogwarts House']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        index = 0
        for stud in x_test:
            (thetaG, thetaR, thetaS, thetaH) = getTheta(stud.shape[0] + 1)
            stud = delete_nan(x_test.shape[1], stud)
            stud = np.insert(stud, 0, 1, axis=1)
            g_score = stud.dot(thetaG)
            r_score = stud.dot(thetaR)
            s_score = stud.dot(thetaS)
            h_score = stud.dot(thetaH)
            #print("score :")
            #print(g_score)
            #print(r_score)
            #print(s_score)
            #print(h_score)
            if max([g_score, r_score, s_score, h_score]) == g_score:
                writer.writerow({'Index': index, 'Hogwarts House': 'Gryffindor'})
            elif max([g_score, r_score, s_score, h_score]) == r_score:
                writer.writerow({'Index': index, 'Hogwarts House': 'Ravenclaw'})
            elif max([g_score, r_score, s_score, h_score]) == s_score:
                writer.writerow({'Index': index, 'Hogwarts House': 'Slytherin'})
            elif max([g_score, r_score, s_score, h_score]) == h_score:
                writer.writerow({'Index': index, 'Hogwarts House': 'Hufflepuff'})
            index += 1

if __name__ == "__main__":
    if len(sys.argv) == 2:
        loader = FileLoader()
        data = loader.load(str(sys.argv[1]))
        predict(data)
    else:
        print("Usage : python logreg_predict.py path_file")