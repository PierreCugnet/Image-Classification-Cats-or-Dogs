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
from keras.preprocessing.image import ImageDataGenerator
import maplotlib.pyplot as plt
import tensorflow as tf


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
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
    'dataset/training_set/training_set',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary')

test_set = test_datagen.flow_from_directory(
    'dataset/test_set/test_set',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary')

classifier.fit(
    training_set,
    steps_per_epoch=(8000/32),
    epochs=25,
    validation_data=test_set,
    validation_steps=(2000/32))


# So we perform a 90% accuracy score.


# Let's try a more complex model : 
# Bigger input shape (128)
# More training (Epochs from 25 to 50)
# 1 more CONV2D+POOLING layer
# 1 more layer in the ANN
INPUT_SHAPE = 128

classifier=Sequential()

#Step 1 - Convolution
classifier.add(Conv2D(32,kernel_size=(3,3), activation='relu', input_shape=(INPUT_SHAPE,INPUT_SHAPE,3)))
classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(Conv2D(32,kernel_size=(3,3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(Conv2D(32,kernel_size=(3,3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(Flatten())

#Step 4 - Full connection= classic ANN
classifier.add(Dense(units= 128, activation='relu'))
classifier.add(Dense(units= 64, activation='relu'))
classifier.add(Dense(units= 1, activation='sigmoid'))
#Compiling the CNN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

# We do not augment test data
test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
    'dataset/training_set/training_set',
    target_size=(INPUT_SHAPE, INPUT_SHAPE),
    batch_size=32,
    class_mode='binary')

test_set = test_datagen.flow_from_directory(
    'dataset/test_set/test_set',
    target_size=(INPUT_SHAPE, INPUT_SHAPE),
    batch_size=32,
    class_mode='binary')

history = classifier.fit(
    training_set,
    steps_per_epoch=(8000/32),
    epochs=50,
    validation_data=test_set,
    validation_steps=(2000/32))

# 

# plot error curves
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

#single prediction

#Put a phto in single_prediction folder with name cat_or_dog_{name} !
def predict():
    import numpy as np
    from keras.preprocessing import image
    PREDICTION_PATH = 'dataset/single_prediction/cat_or_dog_'
    animal_name = input('Whats your monster name\n')
    test_image=image.load_img(PREDICTION_PATH + animal_name +".jpg", target_size=(INPUT_SHAPE,INPUT_SHAPE))
    test_image=image.img_to_array(test_image)
    test_image=np.expand_dims(test_image,axis=0)
    result=classifier.predict(test_image)
    training_set.class_indices
    if result[0][0] == 1 :
        prediction='dog'
    
    else :
        prediction='cat'
    print('{} is a {}'.format(animal_name,prediction))


    
    
    



