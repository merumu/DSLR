import sys
import pickle
from FileLoader import FileLoader
from LogisticRegression import LogisticRegression
import numpy as np
import pandas as pd

def setTheta(thetaG, thetaR, thetaS, thetaH):
    with open('cache', 'wb') as fichier:
        my_pickler = pickle.Pickler(fichier)
        my_pickler.dump(thetaG)
        my_pickler.dump(thetaR)
        my_pickler.dump(thetaS)
        my_pickler.dump(thetaH)
        fichier.close()

def normalize(df):
    df_norm = pd.DataFrame()
    for col in df:
        df_norm[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
    return df_norm

def training(data):
    data = data.dropna()#for later : should drop only NaN line in selected columns
    x_data = data[['Astronomy','Herbology','Defense Against the Dark Arts','Divination','Muggle Studies','Ancient Runes','History of Magic','Transfiguration','Potions','Charms','Flying']]
    #x_data = data[['Arithmancy','Astronomy','Herbology','Defense Against the Dark Arts','Divination','Muggle Studies','Ancient Runes','History of Magic','Transfiguration','Potions','Care of Magical Creatures','Charms','Flying']]
    x_norm = normalize(x_data)
    x_train = x_norm.to_numpy()
    y = np.array(data['Hogwarts House'])
    gryffindor = LogisticRegression("Gryffindor", alpha=0.1, max_iter=1000, verbose=True, learning_rate='constant')
    y_train = np.where(y == 'Gryffindor', 1, 0)
    gryffindor.fit(x_train, y_train)

    ravenclaw = LogisticRegression("Ravenclaw", alpha=0.1, max_iter=1000, verbose=True, learning_rate='constant')
    y_train = np.where(y == 'Ravenclaw', 1, 0)
    ravenclaw.fit(x_train, y_train)

    slytherin = LogisticRegression("Slytherin", alpha=0.1, max_iter=1000, verbose=True, learning_rate='constant')
    y_train = np.where(y == 'Slytherin', 1, 0)
    slytherin.fit(x_train, y_train)

    hufflepuff = LogisticRegression("Hufflepuff", alpha=0.1, max_iter=1000, verbose=True, learning_rate='constant')
    y_train = np.where(y == 'Hufflepuff', 1, 0)
    hufflepuff.fit(x_train, y_train)
    setTheta(gryffindor.thetas, ravenclaw.thetas, slytherin.thetas, hufflepuff.thetas)

if __name__ == "__main__":
    if len(sys.argv) == 2:
        loader = FileLoader()
        data = loader.load(str(sys.argv[1]))
        training(data)
    else:
        print("Usage : python logreg_train.py path_file")
