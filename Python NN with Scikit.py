#-------------------------------------------------------------------------------
# Name:        Scikit
# Purpose:
#
# Author:      Shahazureen Ikwan
#
# Created:     02/08/2019
# Copyright:   (c) ILLEGEAR LAGUNA 2019
# Licence:     <your licence>
#-------------------------------------------------------------------------------
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
iris = load_iris()
print(iris.data[:3])
print(iris.data[15:18])
print(iris.data[37:40])
# we extract only the lengths and widthes of the petals:
X = iris.data[:, (2, 3)]

print(iris.target)

y = (iris.target==0).astype(np.int8)
print(y)

p = Perceptron(random_state=42,
              max_iter=10,
              tol=0.001)
p.fit(X, y)

values = [[1.5, 0.1], [1.8, 0.4], [1.3,0.2]]
for value in X:
    pred = p.predict([value])
    print([pred])


X = [[0., 0.], [0., 1.], [1., 0.], [1., 1.]]
y = [0, 0, 0, 1]
clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(5, 2), random_state=1)
print(clf.fit(X, y))

print("\nweights between input and first hidden layer:")
print(clf.coefs_[0])
print("\nweights between first hidden and second hidden layer:")
print(clf.coefs_[1])

print("\nw0 = ", clf.coefs_[0][0][0])
print("w1 = ", clf.coefs_[0][1][0])

clf.coefs_[0][:,0]

for i in range(len(clf.coefs_)):
    number_neurons_in_layer = clf.coefs_[i].shape[1]
    for j in range(number_neurons_in_layer):
        weights = clf.coefs_[i][:,j]
        print(i, j, weights, end=", ")
        print()
    print()

print("Bias values for first hidden layer:")
print(clf.intercepts_[0])
print("\nBias values for second hidden layer:")
print(clf.intercepts_[1])


result = clf.predict([[0, 0], [0, 1],
                      [1, 0], [0, 1],
                      [1, 1], [2., 2.],
                      [1.3, 1.3], [2, 4.8]])

prob_results = clf.predict_proba([[0, 0], [0, 1],
                                  [1, 0], [0, 1],
                                  [1, 1], [2., 2.],
                                  [1.3, 1.3], [2, 4.8]])
print(prob_results)