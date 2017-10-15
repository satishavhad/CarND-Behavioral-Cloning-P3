# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 18:41:35 2017

@author: avhadsa
"""
import csv

#Read CSV file having image information and steering angel data
samples = []
with open('./driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

#Split data into training and Validaiton sample as 80-20%
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)


import cv2
import numpy as np
import sklearn

#Python generator to generate data for training rather than storing the training data in memory
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = './IMG/'+batch_sample[0].split('\\')[-1]
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                #Add Center images
                images.append(center_image)
                angles.append(center_angle)

                #Add flipped center images
                images.append(cv2.flip(center_image,1))
                angles.append(center_angle*-1.0)

                # create adjusted steering measurements for the side camera images
                correction = 0.1 # this is a parameter to tune
                steering_left = center_angle + correction
                steering_right = center_angle - correction

                # read in images from left and right cameras
                name = './IMG/'+batch_sample[1].split('\\')[-1]
                left_image = cv2.imread(name)
                name = './IMG/'+batch_sample[2].split('\\')[-1]
                right_image = cv2.imread(name)

                #Add left camera images
                images.append(left_image)
                angles.append(steering_left)

                #Add right camera images                
                images.append(right_image)
                angles.append(steering_right)

            # Create NP array of all the images and shuffle the data
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=128)
validation_generator = generator(validation_samples, batch_size=128)

ch, row, col = 3, 160, 320  # image format

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers import Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()

# Preprocess incoming data, centered around zero with small standard deviation 
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(row, col, ch), output_shape=(row, col, ch)))

#Crop the top and bottom of the image
model.add(Cropping2D(cropping=((70,25), (0,0))))

#add 5 convolutional layers
model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(Convolution2D(64, 3, 3, activation="relu"))

#Flatten the image 
model.add(Flatten())
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dense(50))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('relu'))
model.add(Dense(1))

#Compile the model and fit it with 5 epochs
model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch=len(train_samples)*4, validation_data=validation_generator, nb_val_samples=len(validation_samples)*2, nb_epoch=5)

#Save the model to model.h5 file
model.save('model.h5')