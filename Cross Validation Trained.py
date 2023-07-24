import pandas
import numpy
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

# Load dataset
data = "MyFYPDataANN.csv"
nameses = ['S1', 'S2', 'S3','S4','S5','S6','S7','S8','Obstacle']
dataset = pandas.read_csv(data,names = nameses)
array = dataset.values

#Split the data
X = array[:,:8]
Y = array[:,8]

validation_size = 0.25
seed = 10
scoring = 'accuracy'
from sklearn.model_selection import cross_val_score, validation_curve
##from sklearn.model_selection import KFold
kf = KFold(n_splits=4, random_state=seed)
mlp = MLPClassifier(hidden_layer_sizes=(10,10,10), max_iter = 20000)
##

# Train the data
maxcc = 0
for train_indices, test_indices in kf.split(X):
    mlp.fit(X[train_indices], Y[train_indices])
    predictions = mlp.predict (X[test_indices])
    print(accuracy_score(Y[test_indices], predictions))
    print (confusion_matrix(Y[test_indices],predictions))

    cv_results = model_selection.cross_val_score(mlp,X[train_indices], Y[train_indices], cv=kf , scoring = scoring)
    msg = "Kfold: mean: %f std: %f" %(cv_results.mean(), cv_results.std())
    print(msg)

    # Save The data
    #score =(accuracy_score(Y[test_indices], predictions))
    #if score > 0.82 :
    #    testing = 'CrossValue.sav'
    #    pickle.dump (mlp, open (testing,'wb')) # wb = write binary
    #    print('saved')

#print(clf.score(X[test_indices], y[test_indices]))
##predictions = clf.predict (X)
##print(accuracy_score(y, predictions))
##print (confusion_matrix(y,predictions))

##kfold = KFold(n_splits=4,random_state=7)
##cv_results = cross_val_score(clf, X, y, cv=kfold)
##print(cv_results)
##print (cv_results.mean()*100, "%")
##
##predictions = clf.predict (X)
##print(accuracy_score(y, predictions))
##print (confusion_matrix(y,predictions))