"""
Ερώτηση 6: Υλοποιήστε μία παραλλαγή της train_model η οποία να εκπαιδεύει ένα μοντέλο λογιστικής πα-
λινδρόμησης για το ίδιο πρόβλημα. Πως συγκρίνεται η επίδοση αυτού του μοντέλου σε σχέση με το
ΤΝ∆ πολλαπλών επιπέδων;
"""
print(__doc__)

import numpy as np
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import keras
from keras import layers
import seaborn as sns

from assignment1 import load_dataset, evaluate_model

def train_model(X_train, y_train, hparams=None):

    scaler = preprocessing.StandardScaler()
    scaler.fit(X_train)
    m_train = scaler.mean_
    s_train = scaler.scale_
    X_train = scaler.transform(X_train)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    return  model, m_train, s_train

X_train, y_train, X_test, y_test = load_dataset('creditcard.csv')
model, m_train , s_train = train_model(X_train, y_train)
pred, accuracy, precision, recall, f1 = evaluate_model(model, X_test, y_test, m_train, s_train)

print("Accuracy: ", accuracy)
print("Precision: ", precision)
print("Recall: ", recall)
print("F1 score: ", f1)

