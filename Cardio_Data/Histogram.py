import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams


def main():
    df, X, y = load_newdata()
    print(df.columns.values)
    #print(dataset['cardio'].value_counts())

    #histogrm(df)
    #sns.countplot(x='age', hue='cardio', data=df, palette="Set1")
    #sns.countplot(x='gender', hue='cardio', data=df, palette="Set1")
    #sns.countplot(x='height', hue='cardio', data=df, palette="Set1")
    #sns.countplot(x='weight', hue='cardio', data=df, palette="Set1")
    #sns.countplot(x='ap_hi', hue='cardio', data=df, palette="Set1")
    #sns.countplot(x='ap_lo', hue='cardio', data=df, palette="Set1")
    #sns.countplot(x='cholesterol', hue='cardio', data=df, palette="Set1")
    #sns.countplot(x='gluc', hue='cardio', data=df, palette="Set1")
    #sns.countplot(x='smoke', hue='cardio', data=df, palette="Set1")
    #sns.countplot(x='alco', hue='cardio', data=df, palette="Set1")
    #sns.countplot(x='cardio', hue='cardio', data=df, palette="Set1")
    #sns.countplot()
    #plt.show()
    heatmap(df)


def load_dataset():
    df = pd.read_csv('D:\Spring2019\Machine Learning\Project Dataset\Cardio_train\Cardio_train.csv', delimiter=';')
    df.head()
    #print(df['ap_lo'].value_counts())
    df.drop(df[(df['height'] > df['height'].quantile(0.975)) | (df['height'] < df['height'].quantile(0.025))].index,
            inplace=True)
    df.drop(df[(df['weight'] > df['weight'].quantile(0.975)) | (df['weight'] < df['weight'].quantile(0.025))].index,
            inplace=True)
    df.drop(df[(df['ap_hi'] > df['ap_hi'].quantile(0.975)) | (df['ap_hi'] < df['ap_hi'].quantile(0.025))].index,
            inplace=True)
    df.drop(df[(df['ap_lo'] > df['ap_lo'].quantile(0.975)) | (df['ap_lo'] < df['ap_lo'].quantile(0.025))].index,
            inplace=True)

    df.drop('id', axis=1, inplace=True)
    df['age'] = (df['age'] / 365).round().astype('int')
    df.to_csv(r'D:\Spring2019\Machine Learning\Project Dataset\Output\Data.csv')

    # print(dataset.describe())
    # print(dataset.columns.values)
    X = df.iloc[:, 0:11]
    # print(X)
    y = df.iloc[:, 11]

    return df,  X, y
def load_newdata():
    df = pd.read_csv('D:\Spring2019\Machine Learning\Project Dataset\Output\Data.csv')
    df.head()
    X = df.iloc[:, 0:12]
    #print(X)
    y = df.iloc[:, 12]
    #print(y)
    return df, X, y
def heatmap(df):
    corr = df.corr()
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.5,vmin = -0.5, center=0, annot=True,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.show()


def histogrm(df):
    #dataset[['']]
    #dataset[['age']].plot(kind = 'hist', rwidth = 0.8)
    #dataset[['height']].plot(kind='hist', rwidth=0.8)
    #dataset[['gender']].plot(kind='hist',bins = [1,2,3] ,rwidth = 0.8)
    #dataset[['ap_hi']].plot(kind='hist', rwidth=0.8)
    #dataset[['ap_lo']].plot(kind='hist', rwidth=0.8)
    #dataset[['cholesterol']].plot(kind='hist',bins = [1,2,3,4], rwidth=0.8)
    #dataset[['gluc']].plot(kind='hist',bins = [1,2,3,4], rwidth=0.8)
    #dataset[['smoke']].plot(kind='hist',bins = [0,1,2] , rwidth=0.8)
    #dataset[['alco']].plot(kind='hist',bins = [0,1,2] , rwidth=0.8)
    #dataset[['active']].plot(kind='hist',bins = [0,1,2] , rwidth=0.8)
    #dataset[['cardio']].plot(kind='hist', bins = [0,1,2] ,rwidth=0.8)
    #print(dataset['ap_lo'].value_counts())
    plt.show()
#main()
