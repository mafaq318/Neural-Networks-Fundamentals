# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 13:17:15 2023

@author: mafaq
"""

import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from sklearn.metrics import accuracy_score
#%% PART !

detector_output=np.loadtxt('detector_output.dat')
detector_groundtruth=np.loadtxt('detector_groundtruth.dat')



print('Output')
print('length : '+str(len(detector_output)))
print(detector_output)
print('----------------')

print('Ground Truth')
print('length : '+str(len(detector_groundtruth)))
print(detector_groundtruth)
print('----------------')


fpr, tpr, thresholds=metrics.roc_curve(detector_groundtruth,  detector_output)

opt_threshold=thresholds[np.argmax(tpr-fpr)] # Using Youden's J statistic
detector_output_tr=detector_output>opt_threshold

fpr, tpr, thresholds =metrics.roc_curve(detector_groundtruth,  detector_output_tr)

plt.figure()
plt.plot(fpr, tpr)
plt.ylabel('TP Rate')
plt.xlabel('FP Rate')
plt.show()

#%% PART 2

(X_Train, Y_Train),(X_Test, Y_Test)=tf.keras.datasets.fashion_mnist.load_data()

Y_Train_onehot=tf.keras.utils.to_categorical(Y_Train, num_classes=10)
Y_Test_onehot=tf.keras.utils.to_categorical(Y_Test, num_classes=10)

#Normalizing Data
X_Train=X_Train/255.0
X_Test=X_Test/255.0

#Adding Noise
noise_factor=0.2

X_Train_Noise=X_Train + noise_factor * tf.random.normal(shape=X_Train.shape)
X_Test_Noise=X_Test + noise_factor * tf.random.normal(shape=X_Test.shape)

X_Train_Noise = tf.clip_by_value(X_Train_Noise,clip_value_min=0., clip_value_max=1.)
X_Test_Noise = tf.clip_by_value(X_Test_Noise,clip_value_min=0., clip_value_max=1.)

print('X_Train')
print(len(X_Train))
print('Y_Train')
print(len(Y_Train))
print('X_Train')
print(len(X_Test))
print('Y_Train')
print(len(Y_Test))

print('Data Loaded')


##CNN MODEL
model = tf.keras.models.Sequential()

model.add(layers.Conv2D(filters=32, kernel_size=3, strides=1, activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D(pool_size=2))

model.add(layers.Conv2D(filters=64, kernel_size=3, strides=1, activation='relu'))
model.add(layers.MaxPooling2D(pool_size=2))

model.add(layers.Flatten())
model.add(layers.Dense(10, activation='softmax'))

print(model.summary())

model.compile(optimizer='SGD',loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])

fit_model = model.fit(X_Train, Y_Train, epochs=15)

Y_test_pred=model.predict(X_Test)
Y_test_pred = np.argmax(Y_test_pred,axis=1)

Y_test_pred_n=model.predict(X_Test_Noise)
Y_test_pred_n = np.argmax(Y_test_pred_n,axis=1)


Accuracy=accuracy_score(Y_Test, Y_test_pred)
Accuracy_n=accuracy_score(Y_test_pred_n, Y_test_pred)
print(Y_test_pred)
print('Accuracy of our model with clean image is')
print(str(Accuracy*100)+'%')
print('Accuracy of our model with noise image is')
print(str(Accuracy_n*100)+'%')



## AutoEncoder

#Defining the Encoder:
encoder=tf.keras.Sequential(
    [
    tf.keras.layers.Input(shape=(28, 28, 1)),
    tf.keras.layers.Conv2D(filters=16, kernel_size=3, activation='relu', padding='same'),
    tf.keras.layers.Conv2D(filters=8, kernel_size=3, activation='relu', padding='same')
    ]
    )

decoder=tf.keras.Sequential(
    [
    tf.keras.layers.Conv2DTranspose(filters=8, kernel_size=3, activation='relu', padding='same'),
    tf.keras.layers.Conv2DTranspose(filters=16, kernel_size=3, activation='relu', padding='same'),
    tf.keras.layers.Conv2D(1, kernel_size=(3, 3), activation='sigmoid', padding='same')
    ]
    ) 
input = tf.keras.layers.Input(shape=(28, 28, 1))
decoded_img = decoder(encoder(input))
model_cnnautoencoder = tf.keras.Model(inputs=input, outputs=decoded_img)
print(model_cnnautoencoder.summary())

model_cnnautoencoder.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError())
model_cnnautoencoder.fit(X_Train_Noise, X_Train,epochs=15,shuffle=True)

X_Test_denoised=np.array(model_cnnautoencoder.predict(X_Test_Noise))

plt.figure()
plt.subplot(1,2,1)
plt.imshow(tf.squeeze(X_Test_Noise[5]))
plt.xlabel('NoisyImage')
plt.subplot(1,2,2)
plt.imshow(tf.squeeze(X_Test_denoised[5]))
plt.xlabel('Denoised Image')
plt.show()

##
Y_test_pred_autoencoder_dn=model.predict(X_Test_denoised)
Y_test_pred_autoencoder_dn = np.argmax(Y_test_pred_autoencoder_dn,axis=1)
Accuracy=accuracy_score(Y_Test, Y_test_pred_autoencoder_dn)

print(Y_test_pred)
print('Accuracy of our CNNmodel with denoised images from autoencoder is')
print(str(Accuracy*100)+'%')

##

fit_model = model.fit(X_Train_Noise, Y_Train, epochs=15)

Y_test_pred=model.predict(X_Test_Noise)
Y_test_pred = np.argmax(Y_test_pred,axis=1)


Accuracy=accuracy_score(Y_Test, Y_test_pred)

print(Y_test_pred)
print('Accuracy of our model trained on noisy images is')
print(str(Accuracy*100)+'%')

