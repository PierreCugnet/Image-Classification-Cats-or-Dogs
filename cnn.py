# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 09:56:57 2019

@author: Pierre Cugnet
"""

#Part 1 - Building the CNN
from keras.models import Sequential
from keras.layers import Conv2D #if video : 3D (+time)
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

#Initializing the CNN
classifier=Sequential()

#Step 1 - Convolution
classifier.add(Conv2D(32,kernel_size=(3,3), activation='relu', input_shape=(64,64,3)))
#32 feature detectors (cpu issues) of size 3x3 + relu is used to remove linearity ( negative values are zeroed and positive values are kept)
#we need to specify the format of the images because they have
#different formats and because of cpu issues we are going to chose 64x64 and of course 3D arrays because 
#we have colored images RGB and we need to keep that
#stride is equal to 1 by default

#Step 2- Pooling
#size/2 if size is odd size/2 + 1 if size if even then size/2
#goal: reduce the size of feature map --> reduce the processing (time complexity) while keeping the important info ( keeping highest values)
#reducing the time complexity without reducing the performance
classifier.add(MaxPooling2D(pool_size=(2,2)))

#Adding a second convolution layer
classifier.add(Conv2D(32,kernel_size=(3,3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))
#Step 3- Flattening
#All the steps helped us keep the spatial structure information of images
classifier.add(Flatten())

#Step 4 - Full connection= classic ANN
classifier.add(Dense(units= 128, activation='relu'))
classifier.add(Dense(units= 1, activation='sigmoid')) #sigmoid because binary outcome , if categorical then use softmax

#Compiling the CNN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#Part 2 - Fitting the CNN to the images
#Image augmentation is a trick
#that allows us to enrich our image dataset without getting some new images
#the images that we have a rotated, shifted, tweaked etc. and thus as a result we reduce the overfitting problem
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
    'dataset/training_set',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary')

test_set = test_datagen.flow_from_directory(
    'dataset/test_set',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary')

classifier.fit_generator(
    training_set,
    steps_per_epoch=(8000/32),
    epochs=25,
    validation_data=test_set,
    validation_steps=(2000/32))

#single prediction
import numpy as np

from keras.preprocessing import image
test_image=image.load_img('dataset/single_prediction/cat_or_dog_3.jpg', target_size=(64,64))
test_image=image.img_to_array(test_image)
test_image=np.expand_dims(test_image,axis=0)
result=classifier.predict(test_image)
training_set.class_indices
if result[0][0] == 1 :
    prediction='dog'
else :
    prediction='cat'
    



