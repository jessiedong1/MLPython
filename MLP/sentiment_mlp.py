import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.datasets import imdb

(training_data, training_targets), (testing_data, testing_targets) = imdb.load_data(num_words=10000)



def main():
    train_x = vectorize(training_data)
    test_x = vectorize(testing_data)

    test_y = testing_targets
    train_y = training_targets

    # Linear model
    model = models.Sequential()
    model.add(layers.Dense(1, activation="sigmoid", input_shape=(10000,)))
    model.summary()

    model.compile(
        optimizer="rmsprop",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    results = model.fit(
        train_x, train_y,
        epochs=5,
        batch_size=500,
        validation_data=(test_x, test_y)
    )

    # Nonlinear model
    model = models.Sequential()
    # Hidden - Layers
    # model.add(layers.Dense(50, activation = "tanh", input_shape=(10000, )))
    model.add(layers.Dense(50, activation="relu", input_shape=(10000,)))
    # model.add(layers.Dense(50, activation = "sigmoid", input_shape=(10000, )))
    model.add(layers.Dropout(0.3, noise_shape=None, seed=None))
    # Output- Layer
    model.add(layers.Dense(1, activation="sigmoid"))
    model.summary()

    model.compile(
        optimizer="rmsprop",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    results = model.fit(
        train_x, train_y,
        epochs=5,
        batch_size=500,
        validation_data=(test_x, test_y)
    )


def vectorize(sequences, dimension = 10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results


main()