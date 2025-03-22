#!/usr/bin/env python
# coding: utf-8

# In[5]:


import os  
import numpy as np 
import cv2 


# In[6]:


# get_ipython().system('pip install tensorflow')
from tensorflow.keras.utils import to_categorical


# In[7]:


from keras.layers import Input, Dense 
from keras.models import Model


# In[11]:


is_init = False
size = -1

label = []
dictionary = {}
c = 0

for i in os.listdir():
    if i.split(".")[-1] == "npy":
        arr = np.load(i)
        
        print(f"File: {i}, Shape: {arr.shape}")
        if not is_init:
            is_init = True 
            X = arr
            size = X.shape[0]
            y = np.array([i.split('.')[0]] * size).reshape(-1, 1)
        else:
            if len(arr.shape) < 2:  # Check if arr is 1D
                print(f"Error: {i} is not a 2D array. Shape: {arr.shape}")
                continue  # Skip this file
            if arr.shape[1] == X.shape[1]: 
                X = np.concatenate([X, arr], axis=0)
                y = np.concatenate([y, np.array([i.split('.')[0]] * arr.shape[0]).reshape(-1, 1)], axis=0)
			
        label.append(i.split('.')[0])
        dictionary[i.split('.')[0]] = c  
        c = c + 1


for i in range(y.shape[0]):
	y[i, 0] = dictionary[y[i, 0]]
y = np.array(y, dtype="int32")

y = to_categorical(y)

X_new = X.copy()
y_new = y.copy()
counter = 0 

cnt = np.arange(X.shape[0])
np.random.shuffle(cnt)

for i in cnt: 
	X_new[counter] = X[i]
	y_new[counter] = y[i]
	counter = counter + 1



ip = Input(shape=(X.shape[1],))

m = Dense(512, activation="relu")(ip)
m = Dense(256, activation="relu")(m)

op = Dense(y.shape[1], activation="softmax")(m) 

model = Model(inputs=ip, outputs=op)

model.compile(optimizer='rmsprop', loss="categorical_crossentropy", metrics=['acc'])

model.fit(X, y, epochs=50)


model.save("model.h5")
np.save("labels.npy", np.array(label))


# In[ ]:




