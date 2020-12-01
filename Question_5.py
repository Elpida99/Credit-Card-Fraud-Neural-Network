"""
Ερώτηση 5: Υλοποιήστε παραλλαγή των παραπάνω συναρτήσεων η οποία (α) χρησιμοποιεί τυχαίο seed για την
αρχικοποίηση του δικτύου και (β) δημιουργεί τυχαία υποσύνολα εκπαίδευσης και δοκιμής. Για έναν
συγκεκριμένο συνδυασμό υπερπαραμέτρων τρέξτε την εκπαίδευση και την αξιολόγηση 10 φορές και
αναφέρετε το μέσο όρο και την τυπική απόκλιση του F1 score
"""
print(__doc__)

import numpy as np
from sklearn import preprocessing
import keras
from keras import layers

from assignment1 import load_dataset, evaluate_model

##HAPARAMS !!!!!!

def train_model(X_train, y_train, hparams=None):
    scaler = preprocessing.StandardScaler()
    scaler.fit(X_train)
    m_train = scaler.mean_
    s_train = scaler.scale_
    X_train = scaler.transform(X_train)

    rseed = np.random.randint(1000) #random seed for the initialization of weights
    initializer = keras.initializers.RandomNormal(mean=0.0, stddev=1, seed=rseed)
    model = keras.Sequential(
        [
            layers.Dense(20, kernel_initializer=initializer, activation="relu", name="hidden_layer1"),
            layers.Dense(60, kernel_initializer=initializer, activation="relu", name="hidden_layer2"),
            layers.Dense(20, kernel_initializer=initializer, activation="relu", name="hidden_layer3"),
            layers.Dense(1, activation="sigmoid", name="output_layer")
        ]
    )
    if hparams is None:
        hparams = {'epochs': 10}
        opt = "sgd"
    else:
        opt = keras.optimizers.SGD(hparams['learning_rate'])
    model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])

    model.fit(X_train, y_train, epochs=hparams['epochs'], shuffle=False)

    return model, m_train, s_train


f1_score = []
for i in range(10):
    rp_train = np.random.random_sample() #random p_train for splitting the train and test dataset
    print(rp_train)
    X_train, y_train, X_test, y_test = load_dataset('creditcard.csv', rp_train)
    #hparams = {'epochs':10 , 'learning_rate':0.001}
    model, m_train, s_train = train_model(X_train, y_train)
    pred, accuracy, precision, recall, f1 = evaluate_model(model, X_test, y_test, m_train, s_train)
    print(f"f1 is: {f1}")
    f1_score.append(f1)
print(np.mean(f1_score))
print(np.std(f1_score))
