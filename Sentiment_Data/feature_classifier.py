"""
This class is for question 3
Please comment main() in MLP since I imported some functions that I implemented in MLP
"""
import Sentiment_Data.select_features as datasf
from tensorflow.keras import models
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from Sentiment_Data.MLP import *
from numpy import *


def main():
    #Get the processed data
    train_x, train_y, test_x, test_y = load_data()
    train_xf, train_yf, test_xf, test_yf = datasf.fea_datasets()
    #print(train_x.shape)
    #print(train_x1.shape)
    #print(train_y.shape)
    #print(test_x.shape)
    #print(test_y.shape)
    epoch = 5
    batch_size = 500
    predicted_label_lp = load('predict_label_lp.npy')
    predicted_label_mlp = load('predict_label_mlp.npy')

    # get the misclassied
    # print(predicted_label_lp)
    # get the index values of missing classes
    indexs_lp = get_Misclassified(predicted_label_lp, train_yf)
    train_xlp = train_x[indexs_lp, :]
    train_ylp = train_y[indexs_lp]

    indexs_mlp = get_Misclassified(predicted_label_mlp, train_yf)
    train_xmlp = train_x[indexs_mlp, :]
    train_ymlp = train_y[indexs_mlp]


    # Get the result from LP

    result_lp_model_1 = linear_perceptron_model_1(train_xf, train_yf,test_xf, test_yf, epoch, batch_size)

    result_lp_model_2 = linear_perceptron_model_1(train_xlp, train_ylp,test_x, test_y, epoch, batch_size)

    # Get the result from MLP
    result_mlp_model_1 = MLP_model_1(train_xf, train_yf,test_xf, test_yf, epoch, batch_size)
    result_mlp_model_2 = MLP_model_2(train_xmlp, train_ymlp,test_x, test_y, epoch, batch_size)


    # Show the result
    show_result(result_lp_model_1, result_lp_model_2,result_mlp_model_1, result_mlp_model_2)
    #show_result(result_lp_model_1,result_lp_model_2,1,1)
    #show_result(result_mlp_model_1, result_mlp_model_2)




def get_Misclassified(predicted_label_lp , train_yf):
    #print(predicted_label_lp.shape)
    #print(train_yf.shape)
    matches = (predicted_label_lp==train_yf)
    indexs = where(matches == False)
    indexs = indexs[0]
    return indexs

def linear_perceptron_model_1(train_x, train_y,test_x,test_y, epoch, batch_size):
    model = models.Sequential()
    layer = layers.Dense(1, activation="sigmoid", input_shape=(train_x.shape[1],))
    model.add(layer)
    model.summary()

    model.compile(optimizer='nadam', loss='binary_crossentropy', metrics=['accuracy'])

    # Get the loss and accuracy by using cross validation
    results = model.fit(train_x, train_y, epochs=epoch, batch_size=batch_size, validation_data=(test_x,test_y))
    #results = model.predict_proba(test_x)

    #predict_label = model.predict_classes(train_x)
    #predict_label = predict_label.ravel()
    #lp_weights = layer.get_weights()
    #save('predict_label_lp', predict_label)

    return results

    #return predict_label

def linear_perceptron_model_2(train_x, train_y, test_x, test_y, epoch, batch_size):
    model = models.Sequential()
    layer = layers.Dense(1, activation="sigmoid", input_shape=(train_x.shape[1],))
    model.add(layer)
    model.summary()

    model.compile(optimizer='nadam', loss='binary_crossentropy', metrics=['accuracy'])

    # Get the loss and accuracy by using cross validation
    results = model.fit(train_x, train_y, epochs=epoch, batch_size=batch_size,validation_data=(test_x,test_y))
    #results = model.predict_proba(test_x)
    # predict_label = model.predict_classes(train_x)
    # predict_label = predict_label.ravel()
    # lp_weights = layer.get_weights()
    # save('predict_label_lp', predict_label)

    return results


def MLP_model_1(train_x, train_y, test_x, test_y, epoch, batch_size):
    # One hidden layer with 60 neurons
    model = models.Sequential()
    layer = layers.Dense(30, activation="relu", input_shape=(train_x.shape[1],))
    model.add(layer)
    model.add(layers.Dropout(0.5, noise_shape=None, seed=None))
    model.add(layers.Dense(1, activation="sigmoid"))
    model.summary()

    model.compile(optimizer='nadam', loss='binary_crossentropy', metrics=['accuracy'])

# Get the loss and accuracy by using cross validation
    results = model.fit(train_x, train_y, epochs=epoch, batch_size=batch_size, validation_data=(test_x,test_y))
    #results = model.predict_proba(test_x)

    #predict_label = model.predict_classes(train_x)
    #weights_mlp = layer.get_weights()

    #predict_label = model.predict_classes(train_x)
    #predict_label = predict_label.ravel()
    # lp_weights = layer.get_weights()
    #save('predict_label_mlp', predict_label)

    return results

def MLP_model_2(train_x, train_y, test_x, test_y, epoch, batch_size):
    # One hidden layer with 60 neurons
    model = models.Sequential()
    layer = layers.Dense(30, activation="relu", input_shape=(train_x.shape[1],))
    model.add(layer)
    model.add(layers.Dropout(0.5, noise_shape=None, seed=None))
    model.add(layers.Dense(1, activation="sigmoid"))
    model.summary()

    model.compile(optimizer='nadam', loss='binary_crossentropy', metrics=['accuracy'])
    results = model.fit(train_x, train_y, epochs=epoch, batch_size=batch_size,validation_data=(test_x,test_y))
    #predict_results = model.predict_proba(test_x)
    # Get the loss and accuracy by using cross validation
    #results = model.fit(train_x, train_y, epochs=epoch, batch_size=batch_size, validation_data=(test_x, test_y))
    # predict_label = model.predict_classes(train_x)
    # weights_mlp = layer.get_weights()

    # predict_label = model.predict_classes(train_x)
    # predict_label = predict_label.ravel()
    # lp_weights = layer.get_weights()
    # save('predict_label_mlp', predict_label)

    #return results
    return results
    #return predict_label
def show_result(result_lp1,result_lp2, result_mlp1, result_mlp2):
    result_lp1 = result_lp1.history
    result_lp1.keys()
    train_losslp1 =result_lp1['loss']
    val_losslp1 = result_lp1['val_loss']
    train_acclp1 =result_lp1['acc']
    val_acclp1 = result_lp1['val_acc']

    result_lp2 = result_lp2.history
    result_lp2.keys()
    train_losslp2 = result_lp2['loss']
    val_losslp2 = result_lp2['val_loss']
    train_acclp2 = result_lp2['acc']
    val_acclp2 = result_lp2['val_acc']

    train_loss_lp = multiply(train_losslp1,0.4) + multiply(train_losslp2,0.6)
    val_loss_lp = multiply(val_losslp1, 0.4) +  multiply(val_losslp2,0.6)
    train_acc_lp =multiply(train_acclp1, 0.4) +  multiply(train_acclp2,0.6)
    val_acc_lp =multiply(val_acclp1, 0.4)+  multiply(val_acclp2,0.6)


    epochs = range(1,len(train_losslp1)+1)

    plt.subplot(1,2,1)
    plt.plot(epochs, train_loss_lp, 'green', label = 'Training loss')
    plt.plot(epochs, val_loss_lp, 'yellow', label = "Validation loss")
    plt.plot(epochs, train_acc_lp, 'skyblue', label='Training Accuracy')
    plt.plot(epochs, val_acc_lp, 'blue', label="Validation Accuracy")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title("Linear Perceptron Classifier")
    plt.grid(True)
    #plt.legend()
    #plt.show()

    result_mlp1 = result_mlp1.history
    result_mlp1.keys()
    train_lossmlp1 = result_mlp1['loss']
    val_lossmlp1 = result_mlp1['val_loss']
    train_accmlp1 = result_mlp1['acc']
    val_accmlp1 = result_mlp1['val_acc']

    result_mlp2 = result_mlp2.history
    result_mlp2.keys()
    train_lossmlp2 = result_mlp2['loss']
    val_lossmlp2 = result_mlp2['val_loss']
    train_accmlp2 = result_mlp2['acc']
    val_accmlp2 = result_mlp2['val_acc']

    train_loss_mlp = multiply(train_lossmlp1, 0.4) + multiply(train_lossmlp2, 0.6)
    val_loss_mlp = multiply(val_lossmlp1, 0.4) + multiply(val_lossmlp2, 0.6)
    train_acc_mlp = multiply(train_accmlp1, 0.4) + multiply(train_accmlp2, 0.6)
    val_acc_mlp = multiply(val_accmlp1, 0.4) + multiply(val_accmlp2, 0.6)

    plt.subplot(1,2, 2)
    plt.plot(epochs, train_loss_mlp, 'green', label='Training loss')
    plt.plot(epochs, val_loss_mlp, 'yellow', label="Validation loss")
    plt.plot(epochs, train_acc_mlp, 'skyblue', label='Training Accuracy')
    plt.plot(epochs, val_acc_mlp, 'blue', label="Validation Accuracy")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title("MLP Classifier")
    plt.grid(True)

    plt.subplots_adjust(bottom=0.25, top=0.75)
    plt.show()


main()

