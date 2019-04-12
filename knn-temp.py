import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# define column names

# loading training data
df = pd.read_csv('data_features.csv')
df.head()

df = df.values

X = df[0:, 2:]
X = np.asarray(X, 'float')

Y = np.vstack((np.ones((70, 1)), np.zeros((46, 1))))
Y = Y.ravel()

from sklearn import neighbors

accuracy_KNN = np.zeros((10,1))

def KNN(Train, labels, n_neighbors):
    knn = neighbors.KNeighborsClassifier(n_neighbors)
    knn.fit(Train, labels)
    return knn

j = 0

nk = 20

acuracy_k = np.zeros((nk, 1))

for k in range(3, 3+nk):
    accuracy_m = np.zeros((9, 1))
    i = 0
    for M in range(1, 10):    
        X_train, X_valid, label_train, label_valid = train_test_split(X, Y, test_size=0.25, random_state=M)
        knn = KNN(X_train, label_train, k)
        predictions_KNN = knn.predict(X_valid)
        accuracy_m[i]= accuracy_score(label_valid, predictions_KNN)
        i = i + 1
    #    print('knn = ',k,  'accuracy = ', accuracy_KNN[k, 0] * 100, '%')
    acuracy_k[j] = accuracy_m.mean()  
    j = j+1
    
plt.plot(range(3, 3+nk), acuracy_k)
plt.xlabel('k')
plt.ylabel('accuracy')

k = 9
M = 5
X_train, X_valid, label_train, label_valid = train_test_split(X, Y, test_size=0.25, random_state=M)
knn = KNN(X_train, label_train, k)
predictions_KNN = knn.predict(X_valid)
#
def confmat(pred,targets):
        """Confusion matrix"""
        nclasses = 2

        cm = np.zeros((nclasses, nclasses))
        for i in range(nclasses):
            for j in range(nclasses):
                cm[i, j] = np.sum(np.where(pred == i, 1, 0) * np.where(targets == j, 1, 0))

        print("Confusion matrix is:")
        print(cm)
        print("Percentage Correct: ", np.trace(cm) / np.sum(cm) * 100)
        


confmat(predictions_KNN, label_valid)


