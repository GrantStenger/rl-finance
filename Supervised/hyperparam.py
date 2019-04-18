# Import dependencies
import numpy as np
import keras
from keras.layers.core import Dense, Dropout, Activation
from keras.models import Sequential
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform
from keras.utils import np_utils, to_categorical
import tqdm


def data():

    # Constants
    # WINDOW_SIZE = {{choice([5, 10, 25, 50, 100, 200])}}
    WINDOW_SIZE = 25
    NUM_ITERATIONS = 4000
    TRAIN_TEST_RATIO = 0.8

    # Pre-processing
    x = []
    y = []
    window = []

    r_file = open('../data/convert_prices', 'r')
    for i in range(NUM_ITERATIONS):
        window = []
        for j in range(WINDOW_SIZE):
            line = r_file.readline()
            window.append(line[:-1])
        x.append(window)
        next = r_file.readline()[:-1]
        y.append(next)

    r_file.close()

    training_size = int(TRAIN_TEST_RATIO * len(x))

    x_train = x[:training_size]
    x_test = x[training_size:]
    y_train = y[:training_size]
    y_test = y[training_size:]

    # Use Keras to categorize the outputs ("one-hot" vectors)
    y_train_categorical = keras.utils.to_categorical(y_train, num_classes=3)
    y_test_categorical = keras.utils.to_categorical(y_test, num_classes=3)

    return x_train, x_test, y_train_categorical, y_test_categorical


def create_model():

    # Constants
    NUM_HIDDEN_LAYERS = {{choice([0, 1, 2, 4, 8, 16])}}
    NODES_PER_LAYER = {{choice([32, 64, 128, 256, 512])}}
    BATCH_SIZE = {{choice([1, 4, 16, 32, 64, 128])}}
    EPOCHS = {{choice([1, 10, 25, 100])}}

    # Initialize simple neural network model
    model = Sequential()

    # Hidden layer 1: NODES_PER_LAYER neurons, 'relu' activation
    model.add(Dense(units=NODES_PER_LAYER, input_dim=WINDOW_SIZE))
    model.add(Activation('relu'))

    for i in range(NUM_HIDDEN_LAYERS):
        # Hidden layer i: NODES_PER_LAYER neurons, 'relu' activation
        model.add(Dense(units=NODES_PER_LAYER))
        model.add(Activation('relu'))
        model.add(Dropout({{uniform(0, 1)}}))

    # Output layer: 3 neurons (one for each class), softmax activation
    model.add(Dense(units=3))
    model.add(Activation('softmax'))

    # Compile the model
        # Loss: categorical cross-entropy
        # Optimizer: stochastic gradient descent (SGD)
        # Additional metrics: Accuracy
    model.compile(loss='categorical_crossentropy',
                 optimizer={{choice(['rmsprop', 'adam', 'sgd'])}},
                 metrics=['accuracy'])

    # Print model summary
    # model.summary()

    # Fit the model to the training data
    result = model.fit(x_train, y_train_categorical,
         epochs=EPOCHS,
         batch_size=BATCH_SIZE,
         validation_split=0.1,
         verbose=0)

    # Get the highest validation accuracy of the training epochs
    validation_acc = np.amax(result.history['val_acc'])
    print('Best validation acc of epoch:', validation_acc)

    return {'loss': -accuracy, 'status': STATUS_OK, 'model': model}

if __name__ == '__main__':
    best_run, best_model = optim.minimize(model=create_model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=5,
                                          trials=Trials())
    X_train, Y_train, X_test, Y_test = data()
    print("Evalutation of best performing model:")
    print(best_model.evaluate(X_test, Y_test))
    print("Best performing model chosen hyper-parameters:")
    print(best_run)
