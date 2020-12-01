"""
Assignment on the Credit Card Fraud detection dataset, for the
AI applications course

Christos Diou,
Department of Informatics and Telematics
Harokopio University of Athens, 2020
"""

from assignment1 import load_dataset, train_model, evaluate_model

"""
#for question 6:
from assignment1 import load_dataset, evaluate_model
from Question_6 import train_model
"""


X_train, y_train, X_test, y_test = load_dataset('creditcard.csv')
model, m_train, s_train = train_model(X_train, y_train)
pred, accuracy, precision, recall, f1 = evaluate_model(
    model, X_test, y_test, m_train, s_train)

print("On test set with default hyperparameters:")
print("Accuracy: ", accuracy)
print("Precision: ", precision)
print("Recall: ", recall)
print("F1 score: ", f1)

hparams = {'learning_rate': 0.001, 'epochs': 30}

model, m_train, s_train = train_model(X_train, y_train, hparams=hparams)
pred, accuracy, precision, recall, f1 = evaluate_model(
    model, X_test, y_test, m_train, s_train)

print("On test set, with learning rate %f and %d epochs:" %
      (hparams['learning_rate'], hparams['epochs']))
print("Accuracy: ", accuracy)
print("Precision: ", precision)
print("Recall: ", recall)
print("F1 score: ", f1)
