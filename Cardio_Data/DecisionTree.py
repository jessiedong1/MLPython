from sklearn.tree import DecisionTreeClassifier
from Cardio_Data import Histogram as hist
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import confusion_matrix

dataset, x ,y = hist.load_newdata()
#print(dataset.columns.values[0:12])
X_train, X_test, y_train, y_test = train_test_split(x,y,random_state=42)

tree = DecisionTreeClassifier(random_state=0,max_depth=4)
tree.fit(X_train,y_train)
print(tree.score(X_test,y_test))
target_names = [0,1]
pre = tree.predict(X_test)
pre_true = np.array(y_test)
tn, fp, fn, tp = confusion_matrix(pre_true, pre, labels=target_names).ravel()
print('tp {}'.format(tp))
print('fn {}'.format(fn))
print('fp {}'.format(fp))
print('tn {}'.format(tn))






"""

from sklearn.tree import export_graphviz
export_graphviz(tree,out_file='tree.dot', class_names=['No', 'Yes'],feature_names = dataset.columns.values[0:12], impurity=False, filled=True)

import graphviz
with open('tree.dot') as f:
    dot_graph = f.read()

graph = graphviz.Source(dot_graph)
graph.view()

"""

