# -*- coding: utf-8 -*-
"""
Created on Mon May 25 17:51:31 2020

@author: Usuario
"""

import tensorflow as tf
from tensorflow import keras

#%%
mnist=tf.keras.datasets.mnist
#%%
(x_train,y_train),(x_test,y_test)=mnist.load_data()
x_train=tf.keras.utils.normalize(x_train,axis=1)
x_test=tf.keras.utils.normalize(x_test,axis=1)
#%%
model=keras.Sequential()
#%%
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(150,activation='relu'))
model.add(keras.layers.Dense(150,activation='relu'))
model.add(keras.layers.Dense(10,activation='softmax'))
#%%
model.compile('adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
#%%
model.fit(x_train,y_train,epochs=3)
#%%
import matplotlib.pyplot as plt
plt.imshow(x_train[0])
plt.show()
#%%
print(x_train[7])
#%%
import numpy as np
predictions=model.predict([x_test])
print(np.argmax(predictions[10]))
#%%
plt.imshow(x_test[10])