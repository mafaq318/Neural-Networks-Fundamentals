import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
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
model.add(layers.Conv2D(filters=10, kernel_size=3, strides=2, activation='relu', input_shape=(64, 64, 3)))
model.add(layers.MaxPooling2D(pool_size=2))
model.add(layers.Conv2D(filters=10, kernel_size=3, strides=2, activation='relu'))
model.add(layers.MaxPooling2D(pool_size=2))
model.add(layers.Flatten())
model.add(layers.Dense(2, activation='sigmoid'))
print(model.summary())

#%%
model.compile(optimizer='SGD',loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])

fit_model = model.fit(X_train, Y_train, epochs=20,batch_size=32)

plt.plot(fit_model.history['loss'])


Y_test_pred=model.predict(X_test)
Y_test_pred = np.argmax(Y_test_pred,axis=1)

Accuracy=accuracy_score(Y_test, Y_test_pred)
print(Y_test_pred)
print('Accuracy of our model is')
print(str(Accuracy*100)+'%')