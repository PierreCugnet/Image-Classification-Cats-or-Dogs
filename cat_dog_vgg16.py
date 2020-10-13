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
from tensorflow.keras.models import model_from_json
import numpy as np
train_path = 'kaggle/input/training_set/training_set'
# valid_path = 'data/dogs-vs-cats/valid'
test_path = 'kaggle/input/test_set/test_set'

train_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=train_path, target_size=(224,224), classes=['cats', 'dogs'], batch_size=10)
test_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=test_path, target_size=(224,224), classes=['cats', 'dogs'], batch_size=10, shuffle=False)
    
vgg16_model = tf.keras.applications.vgg16.VGG16()


model = Sequential()
for layer in vgg16_model.layers[:-1]:
    model.add(layer)
    
for layer in model.layers:
    layer.trainable = False
    
model.add(Dense(units=2, activation='softmax'))
model.summary()
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

#early_stopping = EarlyStopping(monitor='val_loss', patience=2, verbose=1, mode='auto')  
history = model.fit(x = train_batches, 
          steps_per_epoch = len(train_batches),
          epochs = 1)




predictions = model.predict(x = test_batches, steps = len(test_batches))
cm = confusion_matrix(y_true=test_batches.classes, y_pred=np.argmax(predictions, axis=-1))
cm_plot_labels = ['cat','dog']

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

model.save_weights("model.h5")

#Load the model

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")