import numpy as np
import sys
from FileLoader import FileLoader

def accuracy_score_(y_true, y_pred, normalize=True):
    """
    Compute the accuracy score.
    Args:
        y_true: a scalar or a numpy ndarray for the correct labels
        y_pred: a scalar or a numpy ndarray for the predicted labels
    Returns:
        The accuracy score as a float.
        None on any error.
    Raises:
        This function should not raise any Exception.
    """
    n = 0
    for pred, true in zip(y_pred, y_true):
        if pred == true:
            n += 1
    if normalize == False:
        return n
    if y_pred.shape[0] > 0:
        return n / y_pred.shape[0]
    return None

if __name__ == "__main__":
    if len(sys.argv) == 3:
        loader = FileLoader()
        data1 = loader.load(str(sys.argv[1]))
        data2 = loader.load(str(sys.argv[2]))
        y_true = np.array(data1['Hogwarts House'])
        y_pred = np.array(data2['Hogwarts House'])
        print("score : ", accuracy_score_(y_true, y_pred))
    else:
        print("Usage : python accuracy_score.py path.csv path.csv")
