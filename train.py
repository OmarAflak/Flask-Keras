from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
import numpy as np

def main():
    X = np.array([[0,0],[0,1],[1,0],[1,1]])
    Y = np.array([[0],[1],[1],[0]])

    model = Sequential()
    model.add(Dense(8, input_dim=2, activation='tanh'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=SGD(lr=0.1))
    model.fit(X, Y, batch_size=1, nb_epoch=1000)
    model.summary()

    model.save('xor_model')

if __name__=='__main__':
    main()
