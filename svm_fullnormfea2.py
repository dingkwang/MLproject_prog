# lstm model
import numpy as np
import pandas as pd
from pandas import read_csv
#from keras.models import Sequential
#from keras.layers import Dense
#from keras.layers import Flatten
#from keras.layers import Dropout
#from keras.layers import LSTM
#from keras.utils import to_categorical
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


# loading training data
df = pd.read_csv('normalizedFeatures.csv')
df = df.values

X = df[0:, 1:-1]
X = np.asarray(X, 'float')
#n_timesteps = len(X)
#X = X.reshape(X.shape[0], X.shape[1], 1)

Y = np.zeros(X.shape[0])

for i in range(len(Y)):
    if df[i, -1] == 'prog':
        Y[i] = 1
    
y = Y.ravel()


indices = np.arange(len(y))
trainX, testX, trainy, testy, idxtrain, idxtest= train_test_split(X, y, indices, test_size=0.15, random_state=8)

#from sklearn.svm import SVC 
#from sklearn import svm 
##svclassifier = SVC(kernel='rbf') 
##svclassifier = SVC(kernel='rbf', )
##clf = svm.SVC(gamma='scale', decision_function_shape='ovo')
#clf = svm.SVC(kernel='precomputed')
##svclassifier.fit(trainX, trainy)  
#clf.fit(trainX, trainy) 
#
##y_pred = svclassifier.predict(testX) 
#y_pred = clf.predict(testX)

from sklearn.svm import SVC 
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV

C_range = np.logspace(-2, 10, 13)
gamma_range = np.logspace(-9, 3, 13)
param_grid = dict(gamma=gamma_range, C=C_range)
cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)
grid.fit(X, y)
print("The best parameters are %s with a score of %0.2f"
      % (grid.best_params_, grid.best_score_))

#from sklearn.metrics import classification_report, confusion_matrix  
#print(confusion_matrix(testy,y_pred))  
#print(classification_report(testy,y_pred))  


