"""
Είναι ευαίσθητο το μοντέλο σας στις τιμές των υπερπαραμέτρων;
"""
print(__doc__)

from assignment1 import load_dataset, train_model, evaluate_model

X_train, y_train, X_test, y_test = load_dataset('creditcard.csv')

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

hparams1 = {'learning_rate': 0.1, 'epochs': 50}
model, m_train, s_train = train_model(X_train, y_train, hparams=hparams1)
pred, accuracy, precision, recall, f1 = evaluate_model(
    model, X_test, y_test, m_train, s_train)

print("On test set, with learning rate %f and %d epochs:" %
      (hparams1['learning_rate'], hparams1['epochs']))
print("Accuracy: ", accuracy)
print("Precision: ", precision)
print("Recall: ", recall)
print("F1 score: ", f1)

hparams2 = {'learning_rate': 0.5, 'epochs': 20}
model, m_train, s_train = train_model(X_train, y_train, hparams=hparams2)
pred, accuracy, precision, recall, f1 = evaluate_model(
    model, X_test, y_test, m_train, s_train)

print("On test set, with learning rate %f and %d epochs:" %
      (hparams2['learning_rate'], hparams2['epochs']))
print("Accuracy: ", accuracy)
print("Precision: ", precision)
print("Recall: ", recall)
print("F1 score: ", f1)

hparams3 = {'learning_rate': 1.0, 'epochs': 40}
model, m_train, s_train = train_model(X_train, y_train, hparams=hparams3)
pred, accuracy, precision, recall, f1 = evaluate_model(
    model, X_test, y_test, m_train, s_train)

print("On test set, with learning rate %f and %d epochs:" %
      (hparams3['learning_rate'], hparams3['epochs']))
print("Accuracy: ", accuracy)
print("Precision: ", precision)
print("Recall: ", recall)
print("F1 score: ", f1)

