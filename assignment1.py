import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers


def load_dataset(file, p_train=0.5):
    dataframe = pd.read_csv(file) #read csv file and save to variable "dataframe"

    dataframe = dataframe.drop(['Time'], axis=1) #delete the column with label 'Time'

    X = np.asarray(dataframe.drop(['Class'], axis=1))
    y = np.asarray(dataframe['Class']) #X is all the data except the column 'Class' because that's our target

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=p_train, shuffle=False) #splits the data into the training subset and the test subset accorging to p_train
        # training subset is p_train*data and the rest is test subset

    return X_train, y_train, X_test, y_test


def train_model(X_train, y_train, hparams=None):

    scaler = StandardScaler()
    scaler.fit(X_train) # calculate mean and standard deviation of X_train
    m_train = scaler.mean_ # vector of means used for standardization
    s_train = scaler.scale_ # vector of standard deviations used for standardization
    X_train = scaler.transform(X_train) # stardardize the X_train dataset

    initializer = keras.initializers.RandomNormal(mean=0.0, stddev=1, seed=1234) #random initializer of weights with seed = 1234
    model = keras.Sequential(
        [
            layers.Dense(20, kernel_initializer=initializer, activation="relu", name="hidden_layer1"),
            layers.Dense(60, kernel_initializer=initializer, activation="relu", name="hidden_layer2"),
            layers.Dense(20, kernel_initializer=initializer, activation="relu", name="hidden_layer3"),
            layers.Dense(1, activation="sigmoid", name="output_layer")
        ]
    )

    if hparams is None:
        hparams={'epochs': 10, 'learning_rate': 0.01} #default epochs and learning rate

    model.compile(optimizer=keras.optimizers.SGD(learning_rate=hparams['learning_rate']), loss=keras.losses.BinaryCrossentropy(), metrics=["accuracy"])

    model.fit(X_train, y_train, epochs=hparams['epochs'], shuffle=False)

    return model, m_train, s_train


def evaluate_model(model, X_test, y_test, m, s):

    #standardize X_test using the vectors of means and standard deviations of X_train:
    for col in range(X_test.shape[1]):
        for row in range(X_test.shape[0]):
            X_test[row][col] = (X_test[row][col] - m[col]) / s[col]

    y_pred = (model.predict(X_test) > 0.5).astype("int32") #calculate the predicted y for X_test

    #calculate the accuracy, precision, recall and f1-score of the model:
    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    return y_pred, accuracy, precision, recall, f1
