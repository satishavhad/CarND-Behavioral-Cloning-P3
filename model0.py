# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 18:41:35 2017

@author: avhadsa
"""
import os
import csv

samples = []
with open('./driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

import cv2
import numpy as np
import sklearn

images = []
angles = []
for batch_sample in samples:
    name = './IMG/'+batch_sample[0].split('\\')[-1]
    center_image = cv2.imread(name)
    center_angle = float(batch_sample[3])
    images.append(center_image)
    angles.append(center_angle)

X_train = np.array(images)
y_train = np.array(angles)

#print(images)
print (X_train[0].shape)

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers import Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()

# Preprocess incoming data, centered around zero with small standard deviation 
#model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(ch, row, col), output_shape=(ch, row, col)))
#model.add(Cropping2D(cropping=((70,25), (0,0))))
model.add(Flatten(input_shape=(160, 320, 3)))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, nb_epoch=7, validation_split=0.2, shuffle=True)

model.save('model.h5')