#-------------------------------------------------------------------------------
# Name:        mini project
# Purpose:
#
# Author:      Shahazureen Ikwan
#
# Created:     13/05/2019
# Copyright:   (c) ILLEGEAR LAGUNA 2019
# Licence:     Ohh yeah yeah
#-------------------------------------------------------------------------------

import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Load dataset
ultrasonic = "MyFYPDataANN.csv"
names = ['S1', 'S2', 'S3','S4','S5','S6','S7','S8','Obstacle']
dataset = pandas.read_csv(ultrasonic,names=names)


# Summarize the Dataset
# shape
print(dataset.shape)

# Peek at the data
# head
#print (dataset.head())

# Statistical Summary
# descriptions
print(dataset.describe())

# Class Distribution
print(dataset.groupby('Obstacle').size())

# Unvariable Plots
# box and whisker plots
dataset.plot(kind='box', subplots=True, layout=(4,4), sharex=False)
plt.show()

#histograms
dataset.hist()
plt.show()

# Multivariate Plots
# Scaatter plot matrix
scatter_matrix(dataset)
plt.show()

# Create a validation dataset
# Split-out validation dataset
array = dataset.values
X = array[:,0:8]
Y = array[:,8]
validation_size = 0.25
seed = 10
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X,Y, test_size=validation_size, random_state = seed)

# Test Harness
# Test options and evaluation metric
seed = 7
scoring = 'accuracy'

# Build Models
# Spot check algorithms
models = []
models.append(('LR',LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM',SVC(gamma='auto')))

# evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state = seed)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold , scoring = scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" %(name, cv_results.mean(), cv_results.std())
    print(msg)


# Select Best Model
# Compare Algorithms
fig = plt.figure()
fig.suptitle('Algorith Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()


# Make Predictions
# Make predictions on validation dataset
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

#Make predictions on validation dataset CART
CART = DecisionTreeClassifier()
CART.fit(X_train, Y_train)
predictionsCART = CART.predict(X_validation)
print(accuracy_score(Y_validation, predictionsCART))
print(confusion_matrix(Y_validation, predictionsCART))
print(classification_report(Y_validation, predictionsCART))

#Make predictions on validation dataset NB
NB = GaussianNB()
NB.fit(X_train, Y_train)
predictionsNB = NB.predict(X_validation)
print(accuracy_score(Y_validation, predictionsNB))
print(confusion_matrix(Y_validation, predictionsNB))
print(classification_report(Y_validation, predictionsNB))
