

import os
import random
import numpy as np
import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt



from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import score  


def get_image_info(test_info, dir):
	if dir == 'test':
		images = list(set(list(test_info.image1.unique()) + list(test_info.image2.unique())))
		info = pd.DataFrame(np.array(images).reshape((-1, 1)), columns = ['filename'])
	else:
		info = test_info
	
	info['pixelsx'] = np.nan
	info['pixelsy'] = np.nan
	info['size_bytes'] = np.nan
	
	
	for i in info.index.values:
		try:
			im = Image.open(dir+'/'+info.loc[i, 'filename'])
			info.loc[i, 'pixelsx'], info.loc[i, 'pixelsy'] = im.size
			#im = cv2.imread(dir+'/'+info.loc[i, 'new_filename'])
			#info.loc[i, 'pixelsx'], info.loc[i, 'pixelsy'] = im.shape[0:2]
			info.loc[i, 'size_bytes'] = os.path.getsize(dir+'/'+info.loc[i, 'filename']) 
		
		except:
			print dir+'/'+info.loc[i, 'filename']
		
	return info.rename(columns={'filename' : 'new_filename'})
	

	
def makePairs(trainInfo):
	artists = trainInfo.artist.unique()

	n = trainInfo.groupby('artist').size()
	n = (2*n**2).sum() 
	t = pd.DataFrame(np.zeros((n, 4)), columns=['artist1', 'image1', 'artist2', 'image2'])
	i = 0
	j = 0
	for m in artists:
		
		a = trainInfo[trainInfo.artist==m][['artist', 'new_filename']].values
		use = trainInfo[trainInfo.artist != m].index.values
		np.random.shuffle(use)
		nm = np.min([a.shape[0]**2, trainInfo[trainInfo.artist != m].shape[0] ])
		use = use[0:nm]
		b = trainInfo[trainInfo.artist!=m][['artist', 'new_filename']].ix[use, :].values
		
		a2 = pd.DataFrame(np.concatenate([np.repeat(a[:, 0], a.shape[0]).reshape((-1,1)), np.repeat(a[:, 1], a.shape[0]).reshape((-1,1)), np.tile(a, (a.shape[0], 1))], axis=1), columns=['artist1', 'image1', 'artist2', 'image2'])
		a2 = a2.loc[0:nm, :]
		b2 = pd.DataFrame(np.concatenate([np.tile(a, (a.shape[0], 1))[0:b.shape[0], :], b], axis=1), columns=['artist1', 'image1', 'artist2', 'image2'])
		t.iloc[i:i+a2.shape[0], :] = a2.values
		t.iloc[i+a2.shape[0]:i+a2.shape[0]+b2.shape[0], :] = b2.values
		i += a2.shape[0] +b2.shape[0]
		j += 1
	
	t = t[~t.image2.isin([np.nan, 0])]	
	return t[t.image1 > t.image2]	
	
	
	
def genData(input, split):
	info = input[0]
	data = input[1]
	
	if split=='cv':
		artists = info.artist.unique()
		np.random.shuffle(artists)
		
		info = get_image_info(info, 'train')
		info['bytes_per_pixel'] = 1.0*info['size_bytes']/(info['pixelsx']*info['pixelsy'])
		info['aspect_ratio'] = 1.0*info['pixelsx']/info['pixelsy']	
		train_artists = artists[0:int(0.8*len(artists))]
		test_artists = artists[int(0.8*len(artists)):]	
		
		train = makePairs(info[info.artist.isin(train_artists)])
		test = makePairs(info[info.artist.isin(test_artists)])
		train['in_train'] = True
		test['in_train'] = False
		data = train.append(test)
		data['sameArtist'] = data['artist1'] == data['artist2']
		
	if split=='test':

		info = get_image_info(data, 'test')
		info['bytes_per_pixel'] = 1.0*info['size_bytes']/(info['pixelsx']*info['pixelsy'])
		info['aspect_ratio'] = 1.0*info['pixelsx']/info['pixelsy']	
		
		data['in_train'] = False
	
		if 'artist1' in data.columns:
			data['sameArtist'] = data['artist1'] == data['artist2']

	
	data2 = pd.merge(data, info[['new_filename', 'pixelsx', 'pixelsy', 'size_bytes', 'bytes_per_pixel', 'aspect_ratio']], how='left', left_on='image1', right_on='new_filename')
	data2.drop('new_filename', 1, inplace=True)
	
	data2 = pd.merge(data2, info[['new_filename', 'pixelsx', 'pixelsy', 'size_bytes', 'bytes_per_pixel', 'aspect_ratio']], how='left', left_on='image2', right_on='new_filename')
	data2.drop('new_filename', 1, inplace=True)
	
	xTrain = data2[data2.in_train==True][['pixelsx_x', 'pixelsy_x', 'size_bytes_x', 'bytes_per_pixel_x', 'aspect_ratio_x', 'pixelsx_y', 'pixelsy_y', 'size_bytes_y', 'bytes_per_pixel_y', 'aspect_ratio_y']].values
	xTest = data2[data2.in_train==False][['pixelsx_x', 'pixelsy_x', 'size_bytes_x', 'bytes_per_pixel_x', 'aspect_ratio_x', 'pixelsx_y', 'pixelsy_y', 'size_bytes_y', 'bytes_per_pixel_y', 'aspect_ratio_y']].values
	
	
	if 'artist1' in data.columns: 
		yTrain = data2[data2.in_train==True]['sameArtist'].values
		yTest = data2[data2.in_train==False]['sameArtist'].values
	else:
		yTest = None	
	
	if split=='cv':		
		return xTrain, yTrain, xTest, yTest  
 	if split=='test':
		return xTest, yTest

def train(xTrain, yTrain, xCV, yCV):    
    clf = RandomForestClassifier(n_estimators=100)
    
    clf.fit(xTrain[::5], yTrain[::5])
    print 'Predicting...'
    
    yPrediction = np.zeros(xCV.shape[0])
    for i in xrange(4):
    	yPrediction[i::4] = clf.prob(xCV[i::4])[:,1] 
    
    if not yCV is None:
    	print score(yCV, yPrediction)  
    	
    return yPrediction, clf



def make_submission():
	trainInfo = pd.read_csv('trainInfo.csv')
	submissionInfo = pd.read_csv('submissionInfo.csv')
	print 'prepping training and cv data'
	xTrain, yTrain, xCV, yCV = genData([trainInfo, None], 'cv')	
	print 'prepping test data'
	xTest, yTest = genData([None, submissionInfo], 'test')	
	
	print 'starting classifier'
	yPrediction, clf = train(xTrain, yTrain, xTest, yTest) 

	submission = submissionInfo[['index']]
	submission['sameArtist'] = yPrediction
	submission.to_csv('submission.csv', index=False)



