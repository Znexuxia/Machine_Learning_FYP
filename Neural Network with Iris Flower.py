#-------------------------------------------------------------------------------
# Name:        module2
# Purpose:
#
# Author:      ILLEGEAR LAGUNA
#
# Created:     02/08/2019
# Copyright:   (c) ILLEGEAR LAGUNA 2019
# Licence:     <your licence>
#-------------------------------------------------------------------------------
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn import model_selection
from sklearn.metrics import accuracy_score
import pickle
import pandas
from pandas.plotting import scatter_matrix

# Load dataset
iris = "iris_data.csv"
names = ['sepal-length', 'sepal-width', 'petal-length','petal-width','class']
dataset = pandas.read_csv(iris,names=names)

# Summarize the Dataset
# shape
#print(dataset.shape)

# Peek at the data
# head
#print (dataset.head())

# Statistical Summary
# descriptions
#print(dataset.describe())

# Class Distribution
#print(dataset.groupby('class').size())

# Create Validation dataset
# Split-out validation dataset
array = dataset.values
X = array[:,0:4]
Y = array[:,4]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X,Y, test_size=validation_size, random_state = seed)
#print(X)


mlp = MLPClassifier(hidden_layer_sizes=(10,10), max_iter= 2000)
c = mlp.fit(X_train,Y_train)

# load the predicted data
#loaded_model = pickle.load(open('neural_network.sav', 'rb'))
#result = loaded_model.score(X_validation,Y_validation)
#print(result)

# Make predictions for MLPClassifier
predictions = mlp.predict(X_validation)

# Print the analysis
print (c)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

accuracy = accuracy_score(Y_validation, predictions)
d = float(accuracy)

if d > 0.95:
    #Save the model to disk
    filename = 'neural_network.sav'
    pickle.dump(c,open(filename, 'wb'))


#some time later...
#loaded_model = pickle.load(open(filename, 'rb'))
#result = loaded_model.score(X_validation,Y_validation)
#print(result)

