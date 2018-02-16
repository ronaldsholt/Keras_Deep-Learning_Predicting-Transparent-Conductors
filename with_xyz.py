#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 17:48:14 2018

@author: kra7830
"""

import pandas as pd
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import time
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from spacegroups import load_test, load_train

print("Loading data")
train = load_train()
test = load_test()
print("Done loading data")


# fix random seed for reproducibility
seed = 155
np.random.seed(seed)

#train = pd.read_csv('../df_train.csv')
#test = pd.read_csv('../df_test.csv')


# remove ID from R work, and remove ID colunm from sample... 
train = train.drop('Unnamed: 0.1', axis=1)
train = train.drop('id', axis=1)
test = test.drop('Unnamed: 0.1', axis=1)
test = test.drop('id', axis=1)


train = train[['spacegroup','number_of_total_atoms','percent_atom_al', 'percent_atom_ga', 'percent_atom_in', 'lattice_vector_1_ang', 'lattice_vector_2_ang', 'lattice_vector_3_ang', 'lattice_angle_alpha_degree', 'lattice_angle_beta_degree', 
 'lattice_angle_gamma_degree', 'formation_energy_ev_natom', 'bandgap_energy_ev', 'atomic_densities',
    'vol', 'atomic_densities',
       'avg_mass', 'avg_HOMO', 'avg_LUMO', 'avg_IP', 'avg_rd_max',
       'avg_rs_max', 'avg_rp_max', 'avg_EA', 'avg_Eletronegativity',
       'Ga_0', 'Ga_1', 'Ga_2', 'Ga_3', 'Ga_4', 'Ga_5',
        'Al_0', 'Al_1', 'Al_2', 'Al_3', 'Al_4', 'Al_5', 'O_0',
        'O_1', 'O_2', 'O_3', 'O_4', 'O_5', 'In_0', 'In_1',
        'In_2', 'In_3', 'In_4', 'In_5']]

test = test[['spacegroup','number_of_total_atoms', 'percent_atom_al', 'percent_atom_ga', 'percent_atom_in', 'lattice_vector_1_ang', 'lattice_vector_2_ang', 'lattice_vector_3_ang', 'lattice_angle_alpha_degree', 'lattice_angle_beta_degree', 
 'lattice_angle_gamma_degree', 'atomic_densities', 'vol', 'atomic_densities',
       'avg_mass', 'avg_HOMO', 'avg_LUMO', 'avg_IP', 'avg_rd_max',
       'avg_rs_max', 'avg_rp_max', 'avg_EA', 'avg_Eletronegativity',
       'Ga_0', 'Ga_1', 'Ga_2', 'Ga_3', 'Ga_4', 'Ga_5',
        'Al_0', 'Al_1', 'Al_2', 'Al_3', 'Al_4', 'Al_5', 'O_0',
        'O_1', 'O_2', 'O_3', 'O_4', 'O_5', 'In_0', 'In_1',
        'In_2', 'In_3', 'In_4', 'In_5']]



from sklearn.preprocessing import MinMaxScaler, StandardScaler

t1 = 'formation_energy_ev_natom'
t2 = 'bandgap_energy_ev'

transform_columns = ['number_of_total_atoms','percent_atom_al', 'percent_atom_ga', 'percent_atom_in', 'lattice_vector_1_ang', 'lattice_vector_2_ang', 'lattice_vector_3_ang', 'lattice_angle_alpha_degree', 
                     'lattice_angle_beta_degree', 'lattice_angle_gamma_degree', 'atomic_densities', 'vol', 'atomic_densities','avg_mass', 'avg_HOMO', 'avg_LUMO', 'avg_IP', 'avg_rd_max',
       'avg_rs_max', 'avg_rp_max', 'avg_EA', 'avg_Eletronegativity',
       'Ga_0', 'Ga_1', 'Ga_2', 'Ga_3', 'Ga_4', 'Ga_5',
        'Al_0', 'Al_1', 'Al_2', 'Al_3', 'Al_4', 'Al_5', 'O_0',
        'O_1', 'O_2', 'O_3', 'O_4', 'O_5', 'In_0', 'In_1',
        'In_2', 'In_3', 'In_4', 'In_5']

feature_columns = ['spacegroup'] + transform_columns


# Scaling / Normalizing the data
scaler = MinMaxScaler()
#scaler = StandardScaler()
#scaler.fit(all[transform_columns])
scaler.fit(train[transform_columns])
scaler.fit(test[transform_columns])
train[transform_columns] = scaler.transform(train[transform_columns])
test[transform_columns] = scaler.transform(test[transform_columns])

""" setting up split ratio """
X_train, X_validation = train_test_split(train, test_size=0.20, 
                                         random_state=seed)

y_train = np.log1p(X_train[[t1, t2]]) 
X_train = X_train.drop([t1, t2], axis=1) # Drop the Target Columns

y_validation = np.log1p(X_validation[[t1, t2]])
X_validation = X_validation.drop([ t1, t2], axis=1)

print(X_train.shape, y_train.shape)
print(X_validation.shape, y_validation.shape)

####### Add Custom Metrics
from keras.losses import mean_squared_logarithmic_error
def loss2(y_true, y_pred):
    return tf.sqrt(mean_squared_logarithmic_error(y_true, y_pred))


""" The first run
#############################
##### Building the model ####
############################
np.random.seed(seed)

model = Sequential() # create model
model.add(Dense(20, input_dim=47, activation='relu')) # hidden layer
model.add(Dense(47, activation='relu')) 
#model.add(Dense(47, activation='relu'))#Hidden layer #2
model.add(Dense(16, activation='relu'))#Hidden layer #2
#model.add(Dense(47, activation='relu'))#Hidden layer #2
model.add(Dense(2, activation=None, name='output')) # output layer

#########################
### compile the model ##
####################### 
from keras import optimizers
#sgd = optimizers.
model.compile(loss='mse', 
              optimizer='adam', 
              metrics=['mean_squared_logarithmic_error', 'acc'])

### Try with adam eventually.... 
#my_first_nn.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

# Fitting the model
modeled = model.fit(X_train, y_train, 
                               epochs=100, 
                               verbose=1, 
                               batch_size=10,
                               initial_epoch=0,
                               shuffle=True)        """
from keras.models import Sequential
from keras.layers import Dense, Dropout
#from keras.layers import Embedding
from keras.layers import LSTM

model = Sequential()
model.add(Dense(47, input_dim=47, activation='relu')) # hidden layer
model.add(Dense(128,  activation='relu')) # hidden layer
model.add(Dropout(0.3))
model.add(Dense(20,  activation='relu')) # hidden layer
model.add(Dropout(0.25))
model.add(Dense(47,  activation='relu')) # hidden layer
model.add(Dropout(0.2))
model.add(Dense(20,  activation='relu')) # hidden layer
model.add(Dropout(0.1))
model.add(Dense(10,  activation='relu')) # hidden layer
model.add(Dropout(0.1))
model.add(Dense(128,  activation='relu')) # hidden layer
model.add(Dropout(0.24))
model.add(Dense(20,  activation='relu')) # hidden layer
model.add(Dropout(0.01))
model.add(Dense(2, activation=None, name='output'))

model.compile(loss='mse',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X_train, y_train, batch_size=10, epochs=500)
score = model.evaluate(X_validation, y_validation, batch_size=16)


#Evaluating the model
model.evaluate(X_validation, y_validation, verbose=0)

fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.plot(range(len(model.history['accuracy'])), 
         model.history['accuracy'],linestyle='-', 
         color='blue',label='Training', lw=2)
ax2.plot(range(len(model.history['accuracy'])), 
         model.history['accuracy'],linestyle='-', 
         color='blue',label='Training', lw=2)

leg = ax1.legend(bbox_to_anchor=(0.7, 0.9), 
                 loc=2, borderaxespad=0.,fontsize=13)
ax1.set_xticklabels('')
ax2.set_xlabel('# Epochs',fontsize=14)
ax1.set_ylabel('MSLE',fontsize=14)
ax2.set_ylabel('Accuracy',fontsize=14)
plt.show()


####### More evaluation 
# run a linear regression with the X_validation V. X_target 
pred_test = model.predict(X_validation, verbose=0, batch_size=32)
plt.plot(pred_test[:,1], y_validation.iloc[:,1], 'o')
plt.plot(pred_test[:,0], y_validation.iloc[:,0], 'o')

pred_y = model.predict(test_data, verbose=0, batch_size=16) #first look
plt.plot(pred_y[:,0], 'o')
plt.plot(pred_y[:,1], 'o')

######## If Ready for Submission ###########
sample = pd.read_csv('../../../Desktop/DS_Modeling_Project/kaggle/semi-conductors/sample_submission.csv')
### Start here, if already imported sample

#########################################
##      Predicting with the model     ###
#########################################
test_data = test
pred_y = model.predict(test_data, verbose=0, batch_size=32)

pred_y = np.expm1(pred_y)

pred_y[pred_y[:, 0] < 0, 0] = 0
pred_y[pred_y[:, 1] < 0, 1] = 0

subm = pd.DataFrame()
subm['id'] = sample['id']
subm[t1] = pred_y[:, 0]
subm[t2] = pred_y[:, 1]
subm.to_csv("subm_keras_relu_4000.csv", index=False)


#######
#######
#######
# loss
def np_loss(y_true, y_pred):
    error1 = np.square(y_true[0] - y_pred[0])
    error2 = np.square(y_true[1] - y_pred[1])
    return np.sqrt((error1+error2)/2)



test_sets = model.predict(X_train, verbose=False, batch_size=32)
scores = [np_loss(y, pred) for y, pred in zip(y_train.as_matrix(),
                                                  test_sets)]
final_loss = np.average(scores)
print("Loss from numpy = {}".format(final_loss))



def rmsle(actual, predicted):
    """
    Args:
        actual (1d-array) - array of actual values (float)
        predicted (1d-array) - array of predicted values (float)
    Returns:
        root mean square log error (float)
    """
    return np.sqrt(np.mean(np.power(np.log1p(actual)-np.log1p(predicted), 2)))



pred_test = model.predict(X_validation, verbose=0, batch_size=60)
    
rmsle(pred_test[:,1], y_validation.iloc[:,1])   
rmsle(pred_test[:,0], y_validation.iloc[:,0])   
    
    
    
    
    
    
    
    