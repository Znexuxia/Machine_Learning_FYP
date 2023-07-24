import pandas
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
from pandas.plotting import scatter_matrix
import scikitplot as skplt
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import label_binarize
from sklearn import model_selection
from sklearn.metrics import r2_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.multiclass import OneVsRestClassifier
import numpy as np
import pickle

# Load dataset
data = "MyFYPDataANN.csv"
names = ['S1', 'S2', 'S3','S4','S5','S6','S7','S8','Obstacle']
dataset = pandas.read_csv(data,names=names)

# Load dataset Trained by MLPClassifier
loaded_model = pickle.load(open('CrossValue.sav', 'rb'))

#Split the data
array = dataset.values
X = array[:,0:8]
Y = array[:,8]
validation_size = 0.25
seed = 10
scoring = 'accuracy'
X_train, X_validation, Y_train, Y_validation = train_test_split(X,Y, test_size=validation_size, random_state = seed)
print(dataset)

# Make Predictions
prediction = loaded_model.predict(X)
probs = loaded_model.predict_proba(X)
#print(probs)
# Analysis
#print(accuracy_score(Y,prediction))
#print(confusion_matrix(Y, prediction))
print(classification_report(Y,prediction))

# For graph data
plt.scatter(Y,prediction)
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.show()

# ROC
def multiclass_roc_auc_score(y_test, y_pred, average="macro"):
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    y_pred = lb.transform(y_pred)

    return roc_auc_score(y_test, y_pred, average=average)

c = multiclass_roc_auc_score(Y,prediction,average="macro")

# Roc Graph
skplt.metrics.plot_roc_curve(Y, probs)
plt.show()

# For confusion Matrix
def plot_confusion_matrix(Y, prediction, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(Y, prediction)
    # Only use the labels that appear in the data
    #classes = classes[unique_labels(Y_validation, prediction)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plot_confusion_matrix(Y, prediction, classes=[0,1,2],
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plot_confusion_matrix(Y, prediction, classes=[0,1,2], normalize=True,
                      title='Normalized confusion matrix')

plt.show()
