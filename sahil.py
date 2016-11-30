import pandas as pd
import numpy as np
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.layers.convolutional import Convolution1D
from keras.wrappers.scikit_learn import KerasRegressor
from keras.layers.pooling import MaxPooling1D
import keras
from keras.layers.core import Dropout 
from keras.layers.core import Flatten,Activation
from keras.regularizers import l2
# gensim modules
from gensim import utils
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec
from keras.models import model_from_json
import numpy
import os
# random shuffle
from random import shuffle


# classifier
from sklearn.linear_model import LogisticRegression

import logging
import sys

model = Doc2Vec.load('./imdb.d2v')
train_arrays = numpy.zeros((4000, 100))
train_labels = numpy.zeros(4000)

for i in range(2000):
    prefix_train_pos = 'TRAIN_POS_' + str(i)
    prefix_train_neg = 'TRAIN_NEG_' + str(i)
    train_arrays[i] = model.docvecs[prefix_train_pos]
    train_arrays[2000 + i] = model.docvecs[prefix_train_neg]
    train_labels[i] = 1
    train_labels[2000 + i] = 0
print len(train_arrays[0])
#print train_labels

test_arrays = numpy.zeros((798, 100))
test_labels = numpy.zeros(798)

for i in range(399):
    prefix_test_pos = 'TEST_POS_' + str(i)
    prefix_test_neg = 'TEST_NEG_' + str(i)
    test_arrays[i] = model.docvecs[prefix_test_pos]
    test_arrays[399 + i] = model.docvecs[prefix_test_neg]
    test_labels[i] = 1
    test_labels[399 + i] = 0

train_arrays=train_arrays.reshape(-1,1,100)
#train_arrays=test_arrays.reshape(-1,1,100)

print (train_arrays.shape)
''''
#train_arrays=train_arrays.reshape(-1,1,100)
test_arrays=test_arrays.reshape(-1,1,100)
#train_labels
model = Sequential()

model.add(Convolution1D(50, 10, border_mode='same',input_shape=(1, 100)) ) 
#model.add(Activation('relu'))
#model.add(Convolution1D(10, 10),border_mode='same' )
#model.add(Activation('relu'))
# now model.output_shape == (None, 10, 64)
#model.add(MaxPooling1D(pool_length=(2)))
																													
model.add(Flatten())
model.add(Dense(20))		
model.add(Dropout(0.2))

model.add(Dense(1))		
#model.add(Dropout(0.))
model.add(Activation('sigmoid'))
# add a new conv1d on top
#model.add(Convolution1D(5, 10, border_mode='same'))
#model.add(MaxPooling1D(pool_length=3))
#model.add(Dropout(0.25))
#model.add(Dense(10, activation='sigmoid'))
#model.add(Dense(1))
#model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
		      optimizer='adam',
		      metrics=['accuracy'])
model.fit(train_arrays, train_labels, batch_size=16, nb_epoch=100)

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")
 
# later...
 
# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

loaded_model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
score = loaded_model.evaluate(test_arrays, test_labels, batch_size=16)

#for i in range len(test_arrays):
#	print test_arrays , model.predict(test_arrays
print score , " yes"
'''
'''
log.info('Fitting')
classifier = LogisticRegression()
classifier.fit(train_arrays, train_labels)

LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, penalty='l2', random_state=None, tol=0.0001)

print classifier.score(test_arrays, test_labels)
'''
