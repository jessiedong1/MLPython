"""
The class is built to select the discriminative features
"""
from numpy import *
from Sentiment_Data import MLP


def feature_index():
    lp_weights = load('lp_weights_data.npy')
    #mlp_weights = load('mlp_weights_data.npy')

    lp_weights = lp_weights[0]
    lp_weights = lp_weights.ravel()
    index_lp_weights = argsort(lp_weights)
    #You can slice the desired number of feature
    index_lp_weights = index_lp_weights[9950:10001]
    #print(index_lp_weights)
    #save('index',index_lp_weights)
    return index_lp_weights


def fea_datasets():
    train_x, train_y, test_x, test_y = MLP.load_data()

    # fea_index = feature_index()
    fea_index = load('index.npy')
    train_x = train_x[:, fea_index]
    test_x = test_x[:, fea_index]

    return train_x, train_y, test_x, test_y