# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 11:17:15 2020

@author: Pierre
"""
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
import numpy as np
import pickle
train_path = '/kaggle/input/cat-and-dog/training_set/training_set'
# valid_path = 'data/dogs-vs-cats/valid'
test_path = '../input/cat-and-dog/test_set/test_set'

train_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=train_path, target_size=(224,224), classes=['cats', 'dogs'], batch_size=10)
test_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=test_path, target_size=(224,224), classes=['cats', 'dogs'], batch_size=10, shuffle=False)
    
vgg16_model = tf.keras.applications.vgg16.VGG16()
vgg16_model.summary()

model = Sequential()
for layer in vgg16_model.layers[:-1]:
    model.add(layer)
model.add(Dense(units=2, activation='softmax'))

model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto')  
history = model.fit(x = train_batches, 
          steps_per_epoch = len(train_batches),
          epochs = 5,
          verbose = 2,
          callbacks=[early_stopping]
         )




predictions = model.predict(x = test_batches, steps = len(test_batches), verbose = 0)
cm = confusion_matrix(y_true=test_batches.classes, y_pred=np.argmax(predictions, axis=-1))
cm_plot_labels = ['cat','dog']
plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')

filename = 'cat_dog_vgg16.sav'
pickle.dump(model, open(filename, 'wb'))