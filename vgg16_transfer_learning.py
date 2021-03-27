#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 10:32:45 2021

@author: dhakad
"""

import numpy as np
import os
import time
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing import image
from keras.layers import Dense, Activation, Flatten
from keras.layers import merge, Input
from keras.models import Model
from keras.utils import np_utils
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

# Loading the training data
PATH = os.getcwd()
# Define data path
data_path = PATH + '/data'
data_dir_list = os.listdir(data_path)



img_data_list=[]

for dataset in data_dir_list:
	img_list=os.listdir(data_path+'/'+ dataset)
	print ('Loaded the images of dataset-'+'{}\n'.format(dataset))
	for img in img_list:
		img_path = data_path + '/'+ dataset + '/'+ img
		img = image.load_img(img_path, target_size=(224, 224))
		x = image.img_to_array(img)
		x = np.expand_dims(x, axis=0)
		x = preprocess_input(x)
#		x = x/255
		print('Input image shape:', x.shape)
		img_data_list.append(x)


len(img_data_list)

img_data = np.array(img_data_list)

print(img_data.shape)

img_data = np.rollaxis(img_data,1,0)
print(img_data.shape)

img_data = img_data[0]
print(img_data.shape)


#define the number of classes 
num_classes = 4
num_of_samples = img_data.shape[0]   #808
labels = np.ones((num_of_samples,), dtype='int64')

labels[0:202] = 0
labels[202:404] = 1
labels[404:606] = 2
labels[606:808] = 3

names = ['cats', 'dogs', 'horses', 'humans']



# convert class labels to one-hot encoding
Y = np_utils.to_categorical(labels,num_classes)


# shuffle the dataset
X,y = shuffle(img_data,Y,random_state=2)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=2)


y_train


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#Custom vgg_model       -{VGG16 as fixed feature extractor}
# change into last layer-change 1000class to 4-class layer

image_input = Input(shape=(224,224,3))
model = VGG16(input_tensor=image_input,include_top=True, weights="imagenet")
model.summary()
last_layer = model.get_layer("fc2").output
out = Dense(units=num_classes,activation="softmax", name="output")(last_layer)

custom_vgg_model = Model(image_input,out)
custom_vgg_model.summary()

#freeze all layers except last layer(output layer)
for layer in custom_vgg_model.layers[:-1] :
    layer.trainable = False
    
#custom_vgg_model.layers[21].trainable      -> check perticular 22thlayer is trainable or not

custom_vgg_model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"])

hist = custom_vgg_model.fit(X_train,y_train, batch_size=32,epochs=2, verbose=1, validation_data=(X_test,y_test))

(loss,accuracy) = custom_vgg_model.evaluate(X_test,y_test,batch_size=10,verbose=1)
print("loss : {} % , accuracy:{} %".format(loss*100,accuracy*100))


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# FineTune of VGG16 model

image_input = Input(shape=(224,224,3))
model = VGG16(input_tensor=image_input,include_top=True, weights="imagenet")
model.summary()

last_layer = model.get_layer("block5_pool").output
x = Flatten(name="flatten")(last_layer)
x = Dense(128,activation="relu", name="fc1")(x)
x = Dense(128,activation="relu",name="fc2")(x)
out = Dense(num_classes,activation="softmax", name="output")(x)

custom_vgg_model2 = Model(image_input,out)

#freeze all layers except the dense layers
for layer in custom_vgg_model2.layers[:-3] :
    layer.trainable = False
    
custom_vgg_model2.summary()

custom_vgg_model2.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"])

hist = custom_vgg_model2.fit(X_train,y_train, batch_size=32,epochs=2, verbose=1, validation_data=(X_test,y_test))

(loss,accuracy) = custom_vgg_model2.evaluate(X_test,y_test,batch_size=10,verbose=1)
print("loss : {} % , accuracy:{} %".format(loss*100,accuracy*100))

hist.history.keys()


# plot some graphs
import matplotlib.pyplot as plt
# visualizing losses and accuracy
train_loss=hist.history['loss']
val_loss=hist.history['val_loss']
train_acc=hist.history['accuracy']
val_acc=hist.history['val_accuracy']
xc=range(2)   #beacuse number of epochs = 2

plt.figure(1,figsize=(7,5))
plt.plot(xc,train_loss)
plt.plot(xc,val_loss)
plt.xlabel('num of Epochs')
plt.ylabel('loss')
plt.title('train_loss vs val_loss')
plt.grid(True)
plt.legend(['train','val'])
plt.style.use(['classic'])

plt.figure(2,figsize=(7,5))
plt.plot(xc,train_acc)
plt.plot(xc,val_acc)
plt.xlabel('num of Epochs')
plt.ylabel('accuracy')
plt.title('train_acc vs val_acc')
plt.grid(True)
plt.legend(['train','val'],loc=4)
plt.style.use(['classic'])
