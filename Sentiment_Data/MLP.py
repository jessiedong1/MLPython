"""
This class is for question 2
Please uncomment the main() at the end of the code in order to run it
"""

from numpy import *
from keras.datasets import imdb
from tensorflow.keras import models
from tensorflow.keras import layers
import matplotlib.pyplot as plt

def main():
    train_x, train_y, test_x, test_y= load_data()
    # Get the result for linear perceptron
    epoch = 3
    batch_size = 200
    result_lp, lp_weights= linear_perceptron(train_x, train_y, test_x, test_y, epoch, batch_size)

    # Get the result for MLP
    result_mlp, mlp_weights = MLP(train_x, train_y, test_x, test_y, epoch, batch_size)

    #save('lp_weights_data', lp_weights)
    #save('mlp_weights_data', mlp_weights)

    # Show the result
    show_result(result_lp, result_mlp)


def load_data():
    (train_x, train_y), (test_x, test_y) = imdb.load_data(num_words=10000)
    train_x = vectorize(train_x)
    test_x = vectorize(test_x)


    return train_x, train_y, test_x, test_y

def linear_perceptron(train_x, train_y, test_x, test_y, epoch, batch_size):
    model = models.Sequential()
    layer = layers.Dense(1, activation="sigmoid", input_shape=(train_x.shape[1],))
    model.add(layer)
    model.summary()

#loss=0.5991, acc = 0.7823
    #model.compile(optimizer='sgd', loss = 'binary_crossentropy',metrics=['accuracy'])

#loss = 0.3477, acc = 0.8778
    #model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

#loss = 0.3515, acc = 0.8793
    #model.compile(optimizer='adagrad', loss='binary_crossentropy', metrics=['accuracy'])

#loss = 0.3646, acc = 0.8750
    #model.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])

#loss = 0.3679, acc = 0.8744
    #model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#loss = 0.3778 , acc = 0.8720
    #model.compile(optimizer='adamax', loss='binary_crossentropy', metrics=['accuracy'])

#loss = 0.3204, acc = 0.8825
    model.compile(optimizer='nadam', loss='binary_crossentropy', metrics=['accuracy'])

#Get the loss and accuracy by using cross validation
    results_lp = model.fit(train_x, train_y, epochs=epoch, batch_size=batch_size,validation_data=(test_x, test_y))
    #predict_label = model.predict_classes(train_x)
    lp_weights = layer.get_weights()

    return results_lp, lp_weights


def MLP(train_x, train_y, test_x, test_y, epoch, batch_size):
    # One hidden layer with 60 neurons
    model = models.Sequential()
    layer = layers.Dense(30, activation="relu", input_shape=(train_x.shape[1],))
    model.add(layer)
    model.add(layers.Dropout(0.5, noise_shape=None, seed=None))
    model.add(layers.Dense(1, activation="sigmoid"))
    model.summary()

    model.compile(optimizer='nadam', loss='binary_crossentropy', metrics=['accuracy'])

# Get the loss and accuracy by using cross validation
    results_mlp = model.fit(train_x, train_y, epochs=epoch, batch_size= batch_size, validation_data=(test_x, test_y))
    #predict_label = model.predict_classes(train_x)
    weights_mlp = layer.get_weights()
    return results_mlp,weights_mlp

def vectorize(sequences, dimension = 10000):
    results = zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results

def show_result(result_lp, result_mlp):
    result = result_lp.history
    result.keys()
    train_loss = result['loss']
    val_loss = result['val_loss']
    train_acc = result['acc']
    val_acc = result['val_acc']
    epochs = range(1,len(train_loss)+1)

    plt.subplot(1,2,1)
    plt.plot(epochs, train_loss, 'green', label = 'Training loss')
    plt.plot(epochs, val_loss, 'yellow', label = "Validation loss")
    plt.plot(epochs, train_acc, 'skyblue', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'blue', label="Validation Accuracy")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title("Linear Perceptron Classifier")
    plt.grid(True)
    #plt.legend()
    #plt.show()

    result = result_mlp.history
    result.keys()
    train_loss = result['loss']
    val_loss = result['val_loss']
    train_acc = result['acc']
    val_acc = result['val_acc']
    epochs = range(1, len(train_loss) + 1)

    plt.subplot(1,2, 2)
    plt.plot(epochs, train_loss, 'green', label='Training loss')
    plt.plot(epochs, val_loss, 'yellow', label="Validation loss")
    plt.plot(epochs, train_acc, 'skyblue', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'blue', label="Validation Accuracy")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title("MLP Classifier")
    plt.grid(True)

    plt.subplots_adjust(bottom=0.25, top=0.75)
    plt.show()






#main()
