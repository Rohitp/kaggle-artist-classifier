

from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils

from keras.optimizers import Adam

import Image
from multiprocessing import Pool


def filereader(fname):
	x = np.array(Image.open(fname))
	
	if len(x.shape) == 2:
		#add an additional colour dimension if the only dimensions are width and height
		return preprocess( x.reshape((1, 1) + x.shape) )
	if len(x.shape) == 3:
		return preprocess( x.reshape((1) + x.shape) )	
	
def preprocess(X):
	#this preprocessor crops one pixel along each of the sides of the images
	#return X[:, :, 1:-1, 1:-1] / 255.0	
	return X/ 255.0			


#list(myGenerator(yTrain, bSize, fnames_train))[0]
def myGenerator(y, bSize, fnames):

	#read and preprocess first file to figure out the data dimensions
	sFile = filereader(fnames[0])
	newColors, newRows, newCols = sFile.shape[1:]

	order = np.arange(len(fnames))

	while True:
		
		if not y is None:
			np.random.shuffle(order)
			y = y[order]
			fnames = fnames[order]	
		
		for i in xrange(np.ceil(1.0*len(fnames)/bSize).astype(int)):
			this_bSize = fnames[i*bSize :(i+1)*bSize].shape[0]
			X = np.zeros((this_bSize, newColors, newRows, newCols)).astype('float32')
			
			for i2 in xrange(this_bSize):
				X[i2] = filereader(fnames[i*bSize+i2])
						
			#training set
			if not y is None:	
				yield X, y[i*bSize:(i+1)*bSize]
				
			#test set
			else:
				yield X
					
					



#pred, yTest = fit()
def fit():
	bSize = 16
	epoch = 15

	# input image dimensions
	imgRows, imgCols = 28, 28
	# number of convolutional filters to use
	filters = 32
	# size of pooling area for max pooling
	pools = 2
	# convolution kernel size
	conv = 3

	#load all the labels for the train and test sets
	yTrain = np.loadtxt('labels_train.csv')
	yTest = np.loadtxt('labels_test.csv')
	
	fnames_train = np.array(['train/train'+str(i)+'.png' for i in xrange(len(yTrain))])
	fnames_test = np.array(['test/test'+str(i)+'.png' for i in xrange(len(yTest))])
	
	nb_classes = len(np.unique(yTrain))

	# convert class vectors to binary class matrices
	yTrain = np_utils.to_categorical(yTrain.astype(int), nb_classes)
	yTest = np_utils.to_categorical(yTest.astype(int), nb_classes)

	model = Sequential()

	
	model.add(Convolution2D(filters, conv, conv,
							border_mode='valid', init='he_normal',
							input_shape=(1, imgRows, imgCols)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(pools, pools)))
	model.add(Dropout(0.1))	
	model.add(Convolution2D(filters, conv, conv, border_mode='valid',init='he_normal'))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(pools, pools)))
	model.add(Dropout(0.1))
	

	model.add(Flatten())
	model.add(Dense(32, init='he_normal'))
	model.add(Activation('relu'))
	model.add(Dropout(0.1))	
	model.add(Dense(nb_classes, init='he_normal'))
	model.add(Activation('softmax'))

	optimizer = Adam(lr=1e-3)
	model.compile(loss='categorical_crossentropy',
				  optimizer=optimizer,
				  metrics=['accuracy'])

	model.fit_generator(myGenerator(yTrain, bSize, fnames_train), samples_per_epoch = int(yTrain.shape[0]/bSize), epoch = epoch, verbose=1,callbacks=[], validation_data=None, class_weight=None, max_q_size=10) # show_accuracy=True, nb_worker=1 
		  

	pred = model.predict_generator(myGenerator(None, bSize, fnames_test), len(fnames_test), max_q_size=10) # show_accuracy=True, nb_worker=1 

	#score = model.evaluate(X_test, yTest, verbose=0)
	#print('Test score:', score[0])
	#print('Test accuracy:', score[1])	
	print( 'Test accuracy:', np.mean(np.argmax(pred, axis=1) == np.argmax(yTest, axis=1)) )
	
	return pred, yTest	
		  

