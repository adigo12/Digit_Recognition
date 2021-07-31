import cv2
import numpy as np
import pandas as pd
import random
import os
from sklearn.model_selection import train_test_split

DATADIR = 'extracted_images'
CATEGORIES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '-', '=', 'div']


def get_data_training(path_dir, df):
    df = []
    for i in range(len(CATEGORIES)):
        path = os.path.join(path_dir, CATEGORIES[i])
        category = i
        for img in os.listdir(path):
            img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_COLOR)
            img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
            img_array = cv2.resize(img_array, (28, 28))
            df.append([img_array, category])
    random.shuffle(df)
    return df


def pre_processing(df):
    X = []
    y = []
    for features, label in df:
        X.append(features)
        y.append(label)
    X = np.array(X).reshape([-1, 28, 28, 1])
    X = X.astype('float32') / 255.0
    y = np.array(y)
    return X, y


def store(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.80)
    np.save('X_train.npy', X_train)
    np.save('X_test.npy', X_test)
    np.save('y_train.npy', y_train)
    np.save('y_test.npy', y_test)


df = []
df = get_data_training(DATADIR, df)
X, y = pre_processing(df)
store(X, y)