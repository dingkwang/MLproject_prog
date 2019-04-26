# lstm model
from numpy import mean
from numpy import std
from numpy import dstack
import numpy as np
import pandas as pd
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import LSTM
from keras.utils import to_categorical
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

## load a single file as a numpy array
#def load_file(filepath):
#	dataframe = read_csv(filepath, header=None, delim_whitespace=True)
#	return dataframe.values
#
## load a list of files and return as a 3d numpy array
#def load_group(filenames, prefix=''):
#	loaded = list()
#	for name in filenames:
#		data = load_file(prefix + name)
#		loaded.append(data)
#	# stack group so that features are the 3rd dimension
#	loaded = dstack(loaded)
#	return loaded
#
## load a dataset group, such as train or test
#def load_dataset_group(group, prefix=''):
#	filepath = prefix + group + '/Inertial Signals/'
#	# load all 9 files as a single array
#	filenames = list()
#	# total acceleration
#	filenames += ['total_acc_x_'+group+'.txt', 'total_acc_y_'+group+'.txt', 'total_acc_z_'+group+'.txt']
#	# body acceleration
#	filenames += ['body_acc_x_'+group+'.txt', 'body_acc_y_'+group+'.txt', 'body_acc_z_'+group+'.txt']
#	# body gyroscope
#	filenames += ['body_gyro_x_'+group+'.txt', 'body_gyro_y_'+group+'.txt', 'body_gyro_z_'+group+'.txt']
#	# load input data
#	X = load_group(filenames, filepath)
#	# load class output
#	y = load_file(prefix + group + '/y_'+group+'.txt')
#	return X, y
#
## load the dataset, returns train and test X and y elements
#def load_dataset(prefix=''):
#	# load all train
#	trainX, trainy = load_dataset_group('train', prefix + 'HARDataset/')
#	print(trainX.shape, trainy.shape)
#	# load all test
#	testX, testy = load_dataset_group('test', prefix + 'HARDataset/')
#	print(testX.shape, testy.shape)
#	# zero-offset class values
#	trainy = trainy - 1
#	testy = testy - 1
#	# one hot encode y
#	trainy = to_categorical(trainy)
#	testy = to_categorical(testy)
#	print(trainX.shape, trainy.shape, testX.shape, testy.shape)
#	return trainX, trainy, testX, testy

# fit and evaluate a model
def evaluate_model(trainX, trainy, testX, testy):
	verbose, epochs, batch_size = 0, 15, 16 #epochs = 15 batch size = 64
	n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
#	print(trainX.shape, trainy.shape, testX.shape, testy.shape)
	model = Sequential()
	model.add(LSTM(100, input_shape=(n_timesteps, n_features)))
	model.add(Dropout(0.2))
	model.add(Dense(100, activation='relu'))
	model.add(Dense(n_outputs, activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	# fit network
	model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)
	yhat = model.predict_classes(testX, verbose=0)
	pred = model.predict(testX, batch_size=batch_size, verbose=0)
	print('testy vs pred', np.hstack((testy, pred)))
#	print()
	print('confusion_matrix')
	print(confusion_matrix(testy[:, 1], yhat))
#	incorrects = np.nonzero(model.predict(testX).reshape((-1,))!= testy[:, 1])
#	print('yhat', yhat)	
	# evaluate model
#	_, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)  	
	loss, accuracy = model.evaluate(testX, testy)
	print('accuracy', accuracy)
	incorrects = np.nonzero(yhat != testy[:, 1])
#	pred = model.predict(testX, batch_size=batch_size, verbose=0)
	print('incorrects', incorrects)
#	print('pred', pred) 
	return accuracy, yhat, pred, incorrects
    

## summarize scores
#def summarize_results(scores):
#	print(scores)
#	m, s = mean(scores), std(scores)
#	print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))

# run an experiment
def run_experiment(repeats, X, y):
	# load data
#	trainX, trainy, testX, testy = load_dataset()
	# repeat experiment
 
	scores = list()
	for r in range(repeats):
		indices = np.arange(len(y))
		trainX, testX, trainy, testy, idxtrain, idxtest= train_test_split(X, y, indices, test_size=0.2, random_state=5)
		print('idxtest', idxtest)
		score = evaluate_model(trainX, trainy, testX, testy)
		score = score * 100.0
		print('>#%d: %.3f' % (r+1, score))
		scores.append(score)
	# summarize results
	summarize_results(scores)

def summarize_results(scores):
	print(scores)
	m, s = mean(scores), std(scores)
	print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))

# loading training data
df = pd.read_csv('normalizedFeatures.csv')
df = df.values

X = df[0:, 1:-1]
X = np.asarray(X, 'float')
n_timesteps = len(X)
X = X.reshape(X.shape[0], X.shape[1], 1)

Y = np.zeros(X.shape[0])

for i in range(len(Y)):
    if df[i, -1] == 'prog':
        Y[i] = 1
    
y = Y.ravel()
y = to_categorical(y)

#X = pd.DataFrame(X)
#y = pd.Series(y)
#y = Y.reshape(n_timesteps, 1 , 1 )
M = 5
#trainX, testX, trainy, testy = X, X, y, y
#trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.25, random_state=M)

#accuracy = evaluate_model(trainX, trainy, testX, testy)
#run_experiment(1, X, y)

indices = np.arange(len(y))
trainX, testX, trainy, testy, idxtrain, idxtest= train_test_split(X, y, indices, test_size=0.2, random_state=5)
print('idxtest', idxtest)
accuracy, yhat, pred, incorrects= evaluate_model(trainX, trainy, testX, testy)
print('incorrect index', idxtest[incorrects])