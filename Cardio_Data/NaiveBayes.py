from sklearn.naive_bayes import GaussianNB
from Cardio_Data import Histogram as hist
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from sklearn.metrics import confusion_matrix

dataset, x ,y = hist.load_newdata()
X_train, X_test, y_train, y_test = train_test_split(x,y,random_state=42)

gnb = GaussianNB()
gnb.fit(X_train,y_train)
print(gnb.score(X_test,y_test))




daf = pd.read_csv('D:\Spring2019\Machine Learning\Project Dataset\Output\Dab.csv')
daf.head()
py = daf['Pro_YL']
pn = daf['Pro_NL']
colors = ['navy', 'darkorange']
target_names = [0,1]
label = daf['Pre'].values
y_test1 = np.array(y_test)
print(label)
print(y_test1)

tn, fp, fn, tp = confusion_matrix(y_test1, label, labels=target_names).ravel()
print('tp {}'.format(tp))
print('fn {}'.format(fn))
print('fp {}'.format(fp))
print('tn {}'.format(tn))



def show_results():

    daf = pd.read_csv('D:\Spring2019\Machine Learning\Project Dataset\Output\Dab.csv')
    daf.head()
    py = daf['Pro_YL']
    pn = daf['Pro_NL']
    colors = ['navy', 'darkorange']
    target_names = ['No','Yes']
    labels = daf['Actual']


    std = np.linspace(0, 1, 1000000)


    plt.scatter(py,pn,c=labels, cmap=matplotlib.colors.ListedColormap(colors))
    plt.plot(std, std + 0, linestyle='solid')
    #plt.plot(epochs, val_pre, 'b', label="Test Precision")
    plt.xlabel('Probability_Y')
    plt.ylabel('Probability_N')

    plt.grid(True)
    #plt.legend(loc='best', shadow=False, scatterpoints=1)
    #plt.legend()
    plt.show()

"""
pro = gnb.predict_log_proba(X_test)

pre = gnb.predict(X_test)
print(gnb.class_prior_)
daf = pd.DataFrame(pro,columns=['Pro_N', 'Pro_Y'])
daf['Pre'] = pre
y_test1 = np.array(y_test)
daf['Actual'] = y_test1
print(daf)
py = daf['Pro_Y']
pn = daf['Pro_N']
colors = ['navy', 'darkorange']
target_names = ['No','Yes']
labels = daf['Actual']
"""
#daf.to_csv(r'D:\Spring2019\Machine Learning\Project Dataset\Output\Dab.csv')

