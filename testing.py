# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 00:03:26 2017

@author: avhadsa
"""
import csv

samples = []
with open('./driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)


import cv2
import numpy as np
import sklearn

images = []
angles = []
for batch_sample in samples:
    name = './IMG/'+batch_sample[0].split('\\')[-1]
    center_image = cv2.imread(name)
    cv2.imshow('image0',center_image)
    center_angle = float(batch_sample[3])
    images.append(center_image)
    angles.append(center_angle)
    
    #cv2.imshow('image1',images[0])            
    images.append(cv2.flip(center_image,1))
    angles.append(center_angle*-1.0)
    
    #cv2.imshow('image2',images[1])
    break

# Create NP array
X_train = np.array(images)
y_train = np.array(angles)

#print(images)
print (X_train[0].shape)
#print (X_train[0],X_train[1])
#cv2.imshow('image1', X_train[0])

