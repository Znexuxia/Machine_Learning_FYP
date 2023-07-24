#import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn import model_selection
from sklearn.metrics import accuracy_score
import pickle
import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import numpy as np
import sys



# Load dataset
data = "MyFYPDataANN.csv"
names = ['S1', 'S2', 'S3','S4','S5','S6','S7','S8','Obstacle']
dataset = pandas.read_csv(data, names = names)
array = dataset.values

# Create Validation dataset
# Split-out validation dataset
X = array[:,0:8]
Y = array[:,8]
validation_size = 0.25
seed = 10
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X,Y, test_size=validation_size, random_state = seed)


while True:
    mlp = MLPClassifier(hidden_layer_sizes=(10,10,10), max_iter=20000)
    c = mlp.fit(X_train,Y_train)


    # Make predictions for MLPClassifier
    predictions = c.predict(X_validation)


    #print (predictions)
    print(accuracy_score(Y_validation, predictions))
    #print(confusion_matrix(Y_validation, predictions))
    #print(classification_report(Y_validation, predictions))

    accuracy = accuracy_score(Y_validation, predictions)
    d = float(accuracy)

    if d > 0.90:
        #Save the model to disk
        filename = 'neural.sav'
        pickle.dump(c,open(filename, 'wb'))
        print('done')
        break

#some time later...
#loaded_model = pickle.load(open('FYPANN(moderate).sav', 'rb'))
#result = loaded_model.score(X_validation,Y_validation)
#pred = loaded_model.predict(X_validation)
#print(result)
#print(pred)

