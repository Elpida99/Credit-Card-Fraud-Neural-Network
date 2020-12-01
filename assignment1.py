import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn import preprocessing
import keras
from keras import layers


def load_dataset(file, p_train=0.5):
    data = pd.read_csv(file)
    del data['Time']

    X = np.asarray(data.drop(['Class'], axis=1))
    y = np.asarray(data['Class'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=p_train, shuffle=False)
    return X_train, y_train, X_test, y_test


def train_model(X_train, y_train, hparams=None):

    scaler = preprocessing.StandardScaler()
    scaler.fit(X_train)
    m_train = scaler.mean_
    s_train = scaler.scale_
    X_train = scaler.transform(X_train)

    initializer = keras.initializers.RandomNormal(mean=0.0, stddev=1, seed=1234)
    model = keras.Sequential(
        [
            layers.Dense(20, kernel_initializer=initializer, activation="relu", name="hidden_layer1"),
            layers.Dense(60, kernel_initializer=initializer, activation="relu", name="hidden_layer2"),
            layers.Dense(20, kernel_initializer=initializer, activation="relu", name="hidden_layer3"),
            layers.Dense(1, activation="sigmoid", name="output_layer")
        ]
    )
    if hparams is None:
        hparams={'epochs': 10}
        opt = "sgd"
    else:
        opt = keras.optimizers.SGD(hparams['learning_rate'])
    model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])

    model.fit(X_train, y_train, epochs=hparams['epochs'], shuffle=False)

    return model, m_train, s_train


def evaluate_model(model, X_test, y_test, m, s):

    for col in range(X_test.shape[1]):
        for row in range(X_test.shape[0]):
            X_test[row][col] = (X_test[row][col] - m[col]) / s[col]

    y_pred = (model.predict(X_test) > 0.5).astype("int32")
    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    return y_pred, accuracy, precision, recall, f1
