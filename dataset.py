import numpy as np
import os
from sklearn.utils import shuffle
from utils import cv2HOG, loadImage, loadCutImage


def readImages(folder:str, type:str):
    '''
    读取图像并获取训练所需的 X， Y
    parameters
    ---
    folder: path to  the image data
    type: neg or pos, which needs different preprocess
    '''
    files = os.listdir(folder)
    nums = len(files)
    X = []
    Y = []
    hog = cv2HOG()
    if type == 'pos':
        for filename in files:
            fullpath = os.path.join(folder, filename)
            img = loadImage(fullpath)
            hist = hog.compute(img)
            X.append(hist)
        Y = [1 for i in range(nums)]
    else:
        cnt = 15
        for filename in files:
            fullpath = os.path.join(folder, filename)
            imgs = loadCutImage(fullpath, width=64, height=128, count=cnt)
            hists = [hog.compute(img) for img in imgs]
            X = X + hists
        Y = [0 for i in range(nums*cnt)]
    
    X = np.array(X)
    Y = np.array(Y)

    return X, Y

def loadImgData(train_folder:str, test_folder:str):
    '''
    为加载数据集封装的函数
    parameters
    ---
    train_folder: folder that contains train data
    test_folder: folder that contains test data
    '''
    X0, Y0 = readImages(os.path.join(train_folder, 'neg'), type='neg')
    X1, Y1 = readImages(os.path.join(train_folder, 'pos'), type='pos')
    X = np.concatenate([X0, X1], axis=0)
    Y = np.concatenate([Y0, Y1], axis=0)
    X, Y = shuffle(X, Y, random_state=42)

    _X0, _Y0 = readImages(os.path.join(test_folder, 'neg'), type='neg')
    _X1, _Y1 = readImages(os.path.join(test_folder, 'pos'), type='pos')
    _X = np.concatenate([_X0, _X1], axis=0)
    _Y = np.concatenate([_Y0, _Y1], axis=0)

    _X, _Y = shuffle(_X, _Y, random_state=42)
    print(f"trainX.shape={X.shape}, trainY.shape={Y.shape}")
    print(f"testX.shape={_X.shape}, testY.shape={_Y.shape}")

    return X, Y, _X, _Y


def saveData(data_folder:str, train_x:np.array, train_y:np.array, 
             test_x:np.array, test_y:np.array):
    np.save(os.path.join(data_folder, 'train_x.npy'), train_x)
    np.save(os.path.join(data_folder, 'train_y.npy'), train_y)
    np.save(os.path.join(data_folder, 'test_x.npy'), test_x)
    np.save(os.path.join(data_folder, 'test_y.npy'), test_y)

def loadData(data_folder:str):
    train_x = np.load(os.path.join(data_folder, 'train_x.npy'))
    train_y = np.load(os.path.join(data_folder, 'train_y.npy'))
    test_x = np.load(os.path.join(data_folder, 'test_x.npy'))
    test_y = np.load(os.path.join(data_folder, 'test_y.npy'))

    print(f"trainX.shape={train_x.shape}, trainY.shape={train_y.shape}")
    print(f"testX.shape={test_x.shape}, testY.shape={test_y.shape}")
    
    return train_x, train_y, test_x, test_y