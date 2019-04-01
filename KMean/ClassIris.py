import numpy as np
import pandas as pd

#import data from csv
def main():
    fileName = 'D:\DDownload\ecolicopy.csv'
    k = 3
    max_iteration = 3
    threshold = 0.000001
    num_runs = 3
    # use pandas to read the data
    ad = pd.read_csv(fileName)
    ad.head()
    # put the data into dataframe
    X = pd.DataFrame(ad)
    np.random.seed(200)
    # Initial the mean randomly
    num_samples = X.shape[0]
    num_att = X.shape[1]
    # centers = np.array(7,1)
    centers = []
    for i in range(k):
        centers.append(np.random.randint(0, num_samples - 1))
    cens = X.iloc[[centers[0], centers[1], centers[2]], :]
    X = assignment(k,X,cens)
    print(X.head())

def assignment(k,X, cens):
    for i in range(k):
        # sqrt((x1 - x2)^2 - (y1 - y2)^2)
        X['distance_from_{}'.format(i + 1)] = (
            np.sqrt(
                (X['att1'] - cens.iloc[i, 0]) ** 2
                + (X['att2'] - cens.iloc[i, 1]) ** 2
                + (X['att3'] - cens.iloc[i, 2]) ** 2
                + (X['att4'] - cens.iloc[i, 3]) ** 2
                + (X['att5'] - cens.iloc[i, 4]) ** 2
                + (X['att6'] - cens.iloc[i, 5]) ** 2
                + (X['att7'] - cens.iloc[i, 6]) ** 2

            )
        )
    centroid_distance_cols = ['distance_from_{}'.format(i) for i in range(k)]
    X['closest'] = X.loc[:, centroid_distance_cols].idxmin(axis=1)
    X['closest'] = X['closest'].map(lambda x: int(x.lstrip('distance_from_')))
    return X



main()