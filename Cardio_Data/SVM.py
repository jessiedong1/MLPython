import numpy as np
np.random.seed(333)
from Cardio_Data import Histogram as hist
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

def main():
    train_x, train_y, test_x, test_y= load_data()
    print('SVM')
    # Get the result for linear perceptron
    LinearSVM(train_x, train_y, test_x, test_y)


def load_data():
    dataset, x, y = hist.load_dataset()
    train_x, test_x, train_y, test_y = train_test_split(x, y, random_state=42)

    return train_x, train_y, test_x, test_y
def LinearSVM(X_train,y_train, X_test,y_test):
    clf = SVC(kernel='linear')
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)

    tn, fp, fn, tp= confusion_matrix(y_test, y_pred).ravel()
    precision = 0
    recall = 0
    acc = float((tp + tn) / (tp + fn + fp + tn))
    print("Accuracy: {:.4f}".format(acc))
    if (tp != 0):
        precision = float(tp / (tp + fp))
        recall = float(tp / (tp + fn))
        print("Precision: {:.4f}".format(precision))
        print("Recall: {:.4f}".format(recall))
    print()

    #y_predict = clf.predict(X_test)
    #print(y_predict)
    #clf_predictions = clf.predict(X_test)
    #print("Accuracy: {}%".format(clf.score(X_test, y_test)))
    #print("Accuracy: {}%".format(clf.score(X_test, y_test)))
    return acc, precision, recall
main()
