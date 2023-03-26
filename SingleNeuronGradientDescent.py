# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 10:40:47 2023

@author: mafaq
"""

import numpy as np
from scipy.special import expit # log sig function
import matplotlib.pyplot as plt
#using code snippet from lecture notebook


def MSE(Y_TRUE,Y_PRED):
    difference_square=(Y_TRUE-Y_PRED)**2
    return np.sum(difference_square)/len(Y_TRUE)

#%% creating 50 random data points for each class

X_Hobbit = np.random.normal(1.1,0.3,50)
X_Elf = np.random.normal(1.9,0.4,50)

Y_Hobbit=np.zeros(X_Hobbit.shape)
Y_Elf=np.ones(X_Elf.shape)

plt.figure()
plt.plot(X_Hobbit,Y_Hobbit,'co', label="hobbit")
plt.plot(X_Elf,Y_Elf,'mo', label="elf")
plt.title('Our Dataset')
plt.legend()
plt.xlabel('height [m]')
plt.ylabel('class')
plt.show()


X_data=np.concatenate((X_Hobbit,X_Elf))
Y_data=np.concatenate((Y_Hobbit,Y_Elf))

print('length of dataset')
print(len(X_data))

print('DATASET')
print('X_DATA')
print(X_data)
print('Y_DATA')
print(Y_data)

#%%
#%% Training our data using single neuron

w_0=0
w_1=0
learning_rate=0.5
epochs=5000

sum_y_true=np.sum(Y_data)



for e in range(epochs):
    
    w_0_t=0
    w_1_t=0
    for x_ind, x_val in enumerate(X_data):
        y_pred=expit(w_1*x_val+w_0)
        
        w_1_t+=(Y_data[x_ind]-y_pred)*y_pred*(1-y_pred)*x_val
        w_0_t+=(Y_data[x_ind]-y_pred)*y_pred*(1-y_pred)

    w_1=w_1+(learning_rate/len(Y_data))*w_1_t
    w_0=w_0+(learning_rate/len(Y_data))*w_0_t
    
    loss=MSE(Y_data,expit(w_1*X_data+w_0))
    print('loss= ' + str(loss))
    



#%%


Y_pred=expit(w_1*X_data+w_0)




plt.figure()
plt.plot(X_Hobbit,Y_Hobbit,'co', label="hobbit")
plt.plot(X_Elf,Y_Elf,'mo', label="elf")
plt.plot(X_data,Y_pred,'b-o')
plt.title('Our Dataset')
plt.legend()
plt.xlabel('height [m]')
plt.ylabel('class')
plt.show()

