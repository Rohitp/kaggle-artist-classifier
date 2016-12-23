
from __future__ import absolute_import
from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

import random
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers.core import Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers import Dense, Dropout, Input, Lambda
from keras.optimizers import SGD, RMSprop
from keras import backend as K


def euclidean_distance(vects):
	x, y = vects
	return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))


def eucl_dist_output_shape(shapes):
	shape1, shape2 = shapes
	return shape1


def contrastive_loss(y_true, y_pred):
	'''Contrastive loss from Hadsell-et-al.'06
	http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
	'''
	margin = 1
	return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))


def createPairs(x, dIndice):
	'''Positive and negative pair creation.
	Alternates between positive and negative pairs.
	'''
	pairs = []
	labels = []
	n = min([len(dIndice[d]) for d in range(10)]) - 1
	for d in range(10):
		for i in range(n):
			z1, z2 = dIndice[d][i], dIndice[d][i+1]
			pairs += [[x[z1], x[z2]]]
			inc = random.randrange(1, 10)
			dn = (d + inc) % 10
			z1, z2 = dIndice[d][i], dIndice[dn][i]
			pairs += [[x[z1], x[z2]]]
			labels += [1, 0]
	return np.array(pairs), np.array(labels)



def computAccuracy(predictions, labels):
	'''Compute classification accuracy with a fixed threshold on distances.
	'''
	return labels[predictions.ravel() < 0.5].mean()




def createNetwork(inputD):
	# input image dimensions
	img_colours, imgRows, imgCols = inputD

	# number of convolutional filters to use
	nb_filters = 32
	# size of pooling area for max pooling
	nb_pool = 2
	# convolution kernel size
	nb_conv = 3
	model = Sequential()

	model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
							border_mode='valid',
							input_shape=(img_colours, imgRows, imgCols)))
	model.add(Activation('relu'))
	model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
	#model.add(Dropout(0.1)) #0.25 #too much dropout and loss -> nan

	model.add(Flatten())
	
	model.add(Dense(64, input_shape=(inputD,), activation='relu'))
	#model.add(Dropout(0.05)) #too much dropout and loss -> nan
	model.add(Dense(32, activation='relu'))

	

	return model


		

	
# the data, shuffled and split between train and test sets
(xTrain, yTrain), (xTest, yTest) = mnist.load_data()
#xTrain = xTrain.reshape(60000, 784)
#xTest = xTest.reshape(10000, 784)

# input image dimensions
imgRows, imgCols = 28, 28
xTrain = xTrain.reshape(60000, 1, imgRows, imgCols)
xTest = xTest.reshape(10000, 1, imgRows, imgCols)
	
xTrain = xTrain.astype('float32')
xTest = xTest.astype('float32')
xTrain /= 255
xTest /= 255
#inputD = 784
inputD = (1, imgRows, imgCols)
epoch = 12

# create training+test positive and negative pairs
dIndice = [np.where(yTrain == i)[0] for i in range(10)]
trPairs, trY = createPairs(xTrain, dIndice)

dIndice = [np.where(yTest == i)[0] for i in range(10)]
te_pairs, teY = createPairs(xTest, dIndice)

# network definition
#network = createNetwork(inputD)
network = createNetwork(inputD)

inputA = Input(shape=(1, imgRows, imgCols,))
inputB = Input(shape=(1, imgRows, imgCols,))

# because we re-use the same instance `network`,
# the weights of the network
# will be shared across the two branches
processed_a = network(inputA)
processed_b = network(inputB)

distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([processed_a, processed_b])

model = Model(input=[inputA, inputB], output=distance)

# train
rms = RMSprop()
model.compile(loss=contrastive_loss, optimizer=rms)
model.fit([trPairs[:, 0], trPairs[:, 1]], trY,
		  validation_data=([te_pairs[:, 0], te_pairs[:, 1]], teY),
		  batch_size=128,
		  epoch=epoch)
#model.fit_generator(myGenerator(), samples_per_epoch = 60000, epoch = 2, verbose=2, show_accuracy=True, callbacks=[], validation_data=None, class_weight=None, nb_worker=1)
		  

# compute final accuracy on training and test sets
pred = model.predict([trPairs[:, 0], trPairs[:, 1]])
acc = computAccuracy(pred, trY)
pred = model.predict([te_pairs[:, 0], te_pairs[:, 1]])
tAcc = computAccuracy(pred, teY)

print('* Accuracy on training set: %0.2f%%' % (100 * acc))
print('* Accuracy on test set: %0.2f%%' % (100 * tAcc))