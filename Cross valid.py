import pandas as pd
import matplotlib as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import pickle

# loading dataset
data=pd.read_csv("MyFYPDataANN.csv", header =0)


X = data.iloc [:,0:7].values # all rows and four column (0-3)
y = data.iloc [:,8:10].values # all rows and 5th column
print (X)

from sklearn.model_selection import cross_val_score, validation_curve

##from sklearn.model_selection import KFold
kf = KFold(n_splits=4, random_state=7)
clf = MLPClassifier(hidden_layer_sizes=(30,30,30), max_iter = 7000)
##

maxcc = 0
for train_indices, test_indices in kf.split(X):
    clf.fit(X[train_indices], y[train_indices])

    predictions = clf.predict (X[test_indices])
    print(accuracy_score(y[test_indices], predictions))
    print (confusion_matrix(y[test_indices],predictions))

##    if score > 0.97 :
##        testing = 'testing_dataset.sav'
##        p.dump (mlp, open (testing,'wb')) # wb = write binary

    #print(clf.score(X[test_indices], y[test_indices]))
predictions = clf.predict (X)
print(accuracy_score(y, predictions))
print (confusion_matrix(y,predictions))

##kfold = KFold(n_splits=4,random_state=7)
##cv_results = cross_val_score(clf, X, y, cv=kfold)
##print(cv_results)
##print (cv_results.mean()*100, "%")
##
##predictions = clf.predict (X)
##print(accuracy_score(y, predictions))
##print (confusion_matrix(y,predictions))