import numpy as np
np.random.seed(333)
from keras import models
from keras import layers
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from keras import regularizers
import keras_metrics
from Cardio_Data import Histogram as hist
from sklearn.model_selection import train_test_split

def main():
    train_x, train_y, test_x, test_y= load_data()
    # Get the result for linear perceptron
    epoch = 50
    batch_size = 2000

    # Get the result for MLP
    result_mlp, mlp_weights = MLP(train_x, train_y, test_x, test_y, epoch, batch_size)

    #save('lp_weights_data', lp_weights)
    #save('mlp_weights_data', mlp_weights)

    # Show the result
    show_result(result_mlp)


def load_data():
    dataset, x, y = hist.load_newdata()
    train_x, test_x, train_y, test_y = train_test_split(x, y, random_state=42)

    return train_x, train_y, test_x, test_y




def MLP(train_x, train_y, test_x, test_y, epoch, batch_size):
    # One hidden layer with 60 neurons
    model = models.Sequential()
    layer = layers.Dense(200, activation="relu", input_shape=(train_x.shape[1],),kernel_regularizer=regularizers.l2(1e-10),activity_regularizer=regularizers.l1(1e-10))
    model.add(layer)
    model.add(layers.Dropout(0.1, noise_shape=None, seed=None))
    #model.add(layers.Dropout(0.2, noise_shape=None, seed=None))
    model.add(layers.Dense(100, activation="relu",kernel_regularizer=regularizers.l2(1e-10),activity_regularizer=regularizers.l1(1e-10)))
    model.add(layers.Dropout(0.1, noise_shape=None, seed=None))
    model.add(layers.Dense(60, activation="relu", kernel_regularizer=regularizers.l2(1e-10),
                           activity_regularizer=regularizers.l1(1e-10)))
    model.add(layers.Dense(30, activation="relu", kernel_regularizer=regularizers.l2(1e-10),
                           activity_regularizer=regularizers.l1(1e-10)))
    #model.add(layers.Dropout(0.1, noise_shape=None, seed=None))
    model.add(layers.Dense(10, activation="relu",kernel_regularizer=regularizers.l2(1e-10),activity_regularizer=regularizers.l1(1e-10)))
    model.add(layers.Dense(1, activation="sigmoid"))
    model.summary()
    adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss='binary_crossentropy', metrics=['accuracy', keras_metrics.precision(), keras_metrics.recall()],optimizer=adam)

# Get the loss and accuracy by using cross validation
    results_mlp = model.fit(train_x, train_y, epochs=epoch, batch_size= batch_size, validation_data=(test_x, test_y))
    #predict_label = model.predict_classes(train_x)
    weights_mlp = layer.get_weights()
    return results_mlp,weights_mlp



def show_result(result_lp):
    result = result_lp.history
    result.keys()  # dict_keys(['val_loss', 'val_acc', 'val_precision', 'val_recall', 'loss', 'acc', 'precision', 'recall'])
    # train_loss = result['loss']
    # train_loss.insert(0,0)

    # val_loss = result['val_loss']
    # val_loss.insert(0,0)
    # Accuracy
    train_acc = result['acc']
    train_acc.insert(0, 0)
    val_acc = result['val_acc']
    val_acc.insert(0, 0)
    epochs = range(0, len(val_acc))

    # precision
    train_pre = result['precision']
    train_pre.insert(0, 0)
    val_pre = result['val_precision']
    val_pre.insert(0, 0)

    # recall
    train_recall = result['recall']
    train_recall.insert(0, 0)
    val_recall = result['val_recall']
    val_recall.insert(0, 0)

    plt.title("MLP Classifier")
    plt.subplot(3, 1, 2)
    plt.plot(epochs, train_pre, 'r', label='Train Precision')
    plt.plot(epochs, val_pre, 'b', label="Test Precision")
    plt.xlabel('Epochs')
    plt.ylabel('Precision')
    plt.grid(True)

    plt.subplot(3, 1, 3)

    plt.plot(epochs, train_recall, 'r', label='Train Recall')
    plt.plot(epochs, val_recall, 'b', label="Test Recall")
    plt.xlabel('Epochs')
    plt.ylabel('Recall')
    plt.grid(True)

    plt.subplot(3, 1, 1)
    # plt.plot(epochs, train_loss, 'green', label = 'Training loss')
    # plt.plot(epochs, val_loss, 'yellow', label = "Validation loss")
    plt.plot(epochs, train_acc, 'r', label='Train Accuracy')
    plt.plot(epochs, val_acc, 'b', label="Test Accuracy")
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.grid(True)
    # plt.legend()
    # plt.show()
    # plt.savefig()

    # plt.subplots_adjust(bottom=0.25, top=0.75)
    plt.show()


main()
