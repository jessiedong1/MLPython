import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

colors = ['navy', 'turquoise']
lw = 2
y_names = [0,1]
def main():

    #print(y)
    X,y = load_dataset()
    draw_PCA(X,y)
    #draw_LDA(X,y)

def load_dataset():
    dataset = pd.read_csv('D:\Spring2019\Machine Learning\Project Dataset\Cardio_train\Cardio_train.csv', delimiter=';')
    dataset.head()

    dataset.drop('id', axis=1, inplace=True)
    # print(dataset.describe())
    # print(dataset.columns.values)
    X = dataset.iloc[:, 0:11].values
    # print(X)
    y = dataset.iloc[:, 11].values

    return X,y

def draw_PCA(X,y):

    pca = PCA(n_components=2)

    projected = pca.fit_transform(X)
    #print(projected.shape)
    for color, i, target_name in zip(colors, [0, 1], y_names):
        plt.scatter(projected[y == i, 0], projected[y == i, 1], color=color, alpha=.8, lw=lw,
                    label=target_name)
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title('PCA of Cardio dataset')
    plt.show()
"""
    plt.scatter(projected[:, 0], projected[:, 1],
             edgecolor='none', alpha=0.5,
            cmap=plt.cm.get_cmap('spectral', 10))
    plt.xlabel('component 1')
    plt.ylabel('component 2')

    #plt.show()
"""
def draw_LDA(X,y):
    lda = LDA(n_components=3)
    projected = lda.fit(X, y).transform(X)
    print(projected.shape)


    #for color, i, target_name in zip(colors, [0, 1], y_names):
        #plt.scatter(projected[y == i, 0],projected[y == i, 1], alpha=.8, color=color,
                   # label=target_name)
   # plt.legend(loc='best', shadow=False, scatterpoints=1)
    #plt.title('LDA of IRIS dataset')


main()