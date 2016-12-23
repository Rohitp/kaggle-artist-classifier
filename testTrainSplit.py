
import numpy as np
import os
import Image

np.random.seed(1337)  # for reproducibility



import random
from keras.datasets import mnist

def split():
	(xTrain, yTrain), (xTest, yTest) = mnist.load_data()

	os.mkdir('train')
	os.mkdir('test')
	
	np.savetxt('labels_train.csv', yTrain, header='label')
	np.savetxt('labels_test.csv', yTest, header='label')
	
	for i in xrange(xTrain.shape[0]):
		im = Image.fromarray(np.uint8(xTrain[i]))
		im.save('train'+str(i)+'.png')
	
	for i in xrange(xTest.shape[0]):
		im = Image.fromarray(np.uint8(xTest[i]))
		im.save('test'+str(i)+'.png')	