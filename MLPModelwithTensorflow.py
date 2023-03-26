# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 12:32:09 2023

@author: mafaq
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tqdm import tqdm

#%% Importing and Splitting Data


X_Data=[]
Y_Data=[]

work_path=r'c:\Users\mafaq\Onedrive\Desktop\PatternRecogintion\GTSRB_subset_2\GTSRB_subset_2'

clss=-1
for class_ in tqdm(os.listdir(work_path)):
    clss+=1
    for each in os.listdir(os.path.join(work_path,class_)):
        Y_Data.append(clss)
        X_Data.append(cv2.imread(os.path.join(work_path,class_,each)))

X_Data=np.array(X_Data)
X_Data=X_Data/255.0
Y_Data=np.array(Y_Data)

print('XData')
print(X_Data)
print('XData shape')
print(X_Data.shape)
print('Y_Data')
print(Y_Data)
print('YData shape')
print(Y_Data.shape)


print('Data Load Complete')

X_train, X_test, Y_train, Y_test = train_test_split(X_Data,Y_Data , test_size=0.20,shuffle=True)



print('X_Train')
print(X_train.shape)
print('Y_Train')
print(Y_train.shape)
print('X_Test')
print(X_test.shape)
print('Y_Test')
print(Y_test.shape)


Y_train_onehot = tf.keras.utils.to_categorical(Y_train, num_classes=2)
Y_test_onehot = tf.keras.utils.to_categorical(Y_test, num_classes=2)

#%% making our model

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(64,64,3)))
model.add(tf.keras.layers.Dense(100,activation='relu'))
model.add(tf.keras.layers.Dense(100,activation='relu'))
model.add(tf.keras.layers.Dense(2,activation='softmax'))
print(model.summary())


model.compile(optimizer='SGD',loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])

fit_model = model.fit(X_train, Y_train, epochs=10)

plt.plot(fit_model.history['loss'])


Y_test_pred=model.predict(X_test)
Y_test_pred = np.argmax(Y_test_pred,axis=1)

Accuracy=accuracy_score(Y_test, Y_test_pred)
print(Y_test_pred)
print('Accuracy of our model is')
print(str(Accuracy*100)+'%')