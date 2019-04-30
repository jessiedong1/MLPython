import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs

X, y = make_blobs(n_samples=20,n_features=2,centers = 2,
                  random_state=0, cluster_std=0.80)
print(X.shape)
#plt.scatter(X[:, 0], X[:, 1], c=y, s=50,marker='o',cmap='rainbow')

from sklearn.svm import SVC    # "Support vector classifier"
model = SVC(kernel='rbf',C=100, gamma = 0.1)   #kernel选择线性的
#model = SVC(kernel = 'linear')
model.fit(X, y)



# plt.figure(figsize = (8,5))
# xfit = np.linspace(-1, 3.5)
# plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='winter')

# for m, b, d in [(1, 0.65, 0.33), (0.5, 1.6, 0.55), (-0.2, 2.9, 0.2)]:
#     yfit = m * xfit + b
#     plt.plot(xfit, yfit, )
#     plt.fill_between(xfit, yfit - d, yfit + d, edgecolor='purple',
#                      color='#AAAAAA', alpha=0.5)
#
# plt.xlim(-1, 3.5)

def plot_svc_decision_function(model, ax=None, plot_support=True):
    """Plot the decision function for a 2D SVC"""
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # create grid to evaluate model
    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    Y, X = np.meshgrid(y, x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)

    # plot decision boundary and margins
    ax.contour(X, Y, P, colors='k',
               levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])

    # plot support vectors
    if plot_support:
        ax.scatter(model.support_vectors_[:, 0],
                   model.support_vectors_[:, 1],
                   s=15, linewidth=1, facecolors='red')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)



plt.figure(figsize=(8, 5))
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='winter')
#plt.title('Support Vector Machine' )
plot_svc_decision_function(model)
plt.show()
