"""
Elpida Makri - it21735
Είναι ευαίσθητο το μοντέλο σας στις τιμές των υπερπαραμέτρων;
"""
print(__doc__)

from assignment1 import load_dataset, train_model, evaluate_model

X_train, y_train, X_test, y_test = load_dataset('creditcard.csv')

# hyperparameters with changing number of epochs
e_hparams1 = {'learning_rate': 0.01, 'epochs': 30}
e_hparams2 = {'learning_rate': 0.01, 'epochs': 50}
e_hparams3 = {'learning_rate': 0.01, 'epochs': 70}
e_hparams4 = {'learning_rate': 0.01, 'epochs': 100}
epochs_list = [e_hparams1, e_hparams2, e_hparams3, e_hparams4]

# h yperparameters with changing learning_rate
l_hparams1 = {'learning_rate': 0.005, 'epochs': 20}
l_hparams2 = {'learning_rate': 0.1, 'epochs': 20}
l_hparams3 = {'learning_rate': 0.2, 'epochs': 20}
l_hparams4 = {'learning_rate': 0.3, 'epochs': 20}
learning_rate_list = [l_hparams1, l_hparams2, l_hparams3, l_hparams4]


def sensitivity_analysis(X_train, y_train, X_test, y_test, list):
    for i in range(len(list)):
        model, m_train, s_train = train_model(X_train, y_train, hparams=list[i])
        pred, accuracy, precision, recall, f1 = evaluate_model(model, X_test, y_test, m_train, s_train)

        print("On test set, with learning rate %f and %d epochs:" %
              (list[i]['learning_rate'], list[i]['epochs']))
        print("Accuracy: ", accuracy)
        print("Precision: ", precision)
        print("Recall: ", recall)
        print("F1 score: ", f1)


# sensitivity_analysis(X_train, y_train, X_test, y_test,list=epochs_list)

sensitivity_analysis(X_train, y_train, X_test, y_test, list=learning_rate_list)
