import numpy as np
import pickle
import sklearn.svm as svm
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt



def train(train_x:np.array, train_y:np.array, type:str):
    if type.lower() == 'svc-rbf':
        model = svm.SVC(kernel='rbf', C=600, probability=True, verbose=True)
    elif type.lower() == 'svc-linear':
        model = svm.SVC(kernel='linear', probability=True, verbose=True)
    elif type.lower() == 'svc-poly':
        model = svm.SVC(kernel='poly', degree=3, probability=True, verbose=True)
    else:
        model = svm.SVC(kernel='rbf', C=600, probability=True, verbose=True)

    model.fit(train_x, train_y)
    return model

def saveModel(model, filepath:str):
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)

def loadModel(filepath:str):
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    return model

def makeReport(y_true:np.array, y_pred:np.array):

    cm = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')

    print('Confusion Matrix:')
    print(cm)

    print(f'Accuracy: {acc:.2f}')
    print(f'Precision: {prec:.2f}')
    print(f'Recall: {recall:.2f}')


    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='jet')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()






