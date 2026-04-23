import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier


data = np.loadtxt("Iris.csv", delimiter=",", usecols=[1, 2, 3, 4], skiprows=1)


l1c = np.ones(50, dtype=int)
l2c = np.zeros(50, dtype=int)
target = np.concatenate((l2c, l1c, l2c))

cmp = np.array(['r', 'g'])
plt.figure()
plt.scatter(data[:, 2], data[:, 3], c=cmp[target], s=50, edgecolors='none')

#Quel est votre constat ?
#Les Iris setosa (en rouge) sont clairement séparés des autres classes

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(
    data[:,1], 
    data[:,2], 
    data[:,3],  
    c=[cmp[i] for i in target],
    s=50
)

ax.set_xlabel('Sepal width')
ax.set_ylabel('Petal length')
ax.set_zlabel('Petal width')



np.random.seed(10)

X_train, X_test, y_train, y_test = train_test_split(
    data, target, test_size=0.4
)

clf = MLPClassifier(
    solver='sgd',
    learning_rate_init=0.01,
    max_iter=10000,
    random_state=0
)

clf.fit(X_train[:,2:4], y_train)
print("Train:", clf.score(X_train[:,2:4], y_train))
print("Test:", clf.score(X_test[:,2:4], y_test))


nx, ny = 200, 200

x_min, x_max = plt.xlim()
y_min, y_max = plt.ylim()

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, nx),
    np.linspace(y_min, y_max, ny)
)

Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])
Z1 = Z[:,1].reshape(xx.shape)

plt.contour(xx, yy, Z1, [0.5])
plt.show()