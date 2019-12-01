#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 17:20:03 2019

@author: francesco
"""

import keras 
import keras.backend as K
import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

"""
check if setting z_mean, z_log_var as global variable in autoencoder.py makes a difference
the code below is equivalent to the following parameters (HARDataset):
{'input_shape': (561, 1), 
'n_blocks': 8, 
'latent_dim': 50, 
'encoder_latent_layer_type': 'variational', 
'encoder_args': {'filters': [2, 4, 8, 16, 32, 64, 128, 256], 
'kernel_size': [21, 18, 15, 13, 11, 8, 5, 3], 
'padding': 'same', 
'activation': 'selu', 
'pooling': [1, 1, 1, 1, 1, 1, 1, 1]}}
"""


def sampling(args):
        z_mean, z_log_var = args
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        # by default, random_normal has mean = 0 and std = 1.0
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon
    
input_layer = keras.layers.Input(shape=(561, 1))
blocks = input_layer

blocks = keras.layers.Conv1D(filters=2, kernel_size=21, padding = "same")(blocks)
blocks = keras.layers.normalization.BatchNormalization()(blocks)
blocks = keras.layers.Activation("selu")(blocks)
blocks = keras.layers.MaxPooling1D(1)(blocks)

blocks = keras.layers.Conv1D(filters=4, kernel_size=18, padding = "same")(blocks)
blocks = keras.layers.normalization.BatchNormalization()(blocks)
blocks = keras.layers.Activation("selu")(blocks)
blocks = keras.layers.MaxPooling1D(1)(blocks)

blocks = keras.layers.Conv1D(filters=8, kernel_size=15, padding = "same")(blocks)
blocks = keras.layers.normalization.BatchNormalization()(blocks)
blocks = keras.layers.Activation("selu")(blocks)
blocks = keras.layers.MaxPooling1D(1)(blocks)

blocks = keras.layers.Conv1D(filters=16, kernel_size=13, padding = "same")(blocks)
blocks = keras.layers.normalization.BatchNormalization()(blocks)
blocks = keras.layers.Activation("selu")(blocks)
blocks = keras.layers.MaxPooling1D(1)(blocks)

blocks = keras.layers.Conv1D(filters=32, kernel_size=11, padding = "same")(blocks)
blocks = keras.layers.normalization.BatchNormalization()(blocks)
blocks = keras.layers.Activation("selu")(blocks)
blocks = keras.layers.MaxPooling1D(1)(blocks)

blocks = keras.layers.Conv1D(filters=64, kernel_size=8, padding = "same")(blocks)
blocks = keras.layers.normalization.BatchNormalization()(blocks)
blocks = keras.layers.Activation("selu")(blocks)
blocks = keras.layers.MaxPooling1D(1)(blocks)

blocks = keras.layers.Conv1D(filters=128, kernel_size=5, padding = "same")(blocks)
blocks = keras.layers.normalization.BatchNormalization()(blocks)
blocks = keras.layers.Activation("selu")(blocks)
blocks = keras.layers.MaxPooling1D(1)(blocks)

blocks = keras.layers.Conv1D(filters=256, kernel_size=3, padding = "same")(blocks)
blocks = keras.layers.normalization.BatchNormalization()(blocks)
blocks = keras.layers.Activation("selu")(blocks)
blocks = keras.layers.MaxPooling1D(1)(blocks)

#{'filters': [2, 4, 8, 16, 32, 64, 128, 256], 'kernel_size': [21, 18, 15, 13, 11, 8, 5, 3], 'padding': 'same', 'activation': 'selu', 'pooling': [1, 1, 1, 1, 1, 1, 1, 1]}}

blocks = keras.layers.Conv1D(filters=1, kernel_size=1, padding = "same")(blocks)
blocks = keras.layers.Activation('linear')(blocks)
blocks = keras.layers.Flatten()(blocks)

latent_dim = 50
z_mean = keras.layers.Dense(latent_dim, name='z_mean')(blocks)
z_log_var = keras.layers.Dense(latent_dim, name='z_log_var')(blocks)
z = keras.layers.Lambda(sampling, output_shape = (latent_dim,), name='z')([z_mean, z_log_var])
    
blocks = z

blocks = keras.layers.Dense(561)(blocks)
blocks = keras.layers.Reshape((561, 1))(blocks)   

blocks = keras.layers.UpSampling1D(size=1)(blocks)
blocks = keras.layers.Conv1D(filters=256, kernel_size=3, padding = "same")(blocks)
blocks = keras.layers.normalization.BatchNormalization()(blocks)
blocks = keras.layers.Activation("selu")(blocks)

blocks = keras.layers.UpSampling1D(size=1)(blocks)
blocks = keras.layers.Conv1D(filters=128, kernel_size=5, padding = "same")(blocks)
blocks = keras.layers.normalization.BatchNormalization()(blocks)
blocks = keras.layers.Activation("selu")(blocks)

blocks = keras.layers.UpSampling1D(size=1)(blocks)
blocks = keras.layers.Conv1D(filters=64, kernel_size=8, padding = "same")(blocks)
blocks = keras.layers.normalization.BatchNormalization()(blocks)
blocks = keras.layers.Activation("selu")(blocks)

blocks = keras.layers.UpSampling1D(size=1)(blocks)
blocks = keras.layers.Conv1D(filters=32, kernel_size=11, padding = "same")(blocks)
blocks = keras.layers.normalization.BatchNormalization()(blocks)
blocks = keras.layers.Activation("selu")(blocks)

blocks = keras.layers.UpSampling1D(size=1)(blocks)
blocks = keras.layers.Conv1D(filters=16, kernel_size=13, padding = "same")(blocks)
blocks = keras.layers.normalization.BatchNormalization()(blocks)
blocks = keras.layers.Activation("selu")(blocks)

blocks = keras.layers.UpSampling1D(size=1)(blocks)
blocks = keras.layers.Conv1D(filters=8, kernel_size=15, padding = "same")(blocks)
blocks = keras.layers.normalization.BatchNormalization()(blocks)
blocks = keras.layers.Activation("selu")(blocks)

blocks = keras.layers.UpSampling1D(size=1)(blocks)
blocks = keras.layers.Conv1D(filters=4, kernel_size=18, padding = "same")(blocks)
blocks = keras.layers.normalization.BatchNormalization()(blocks)
blocks = keras.layers.Activation("selu")(blocks)

blocks = keras.layers.UpSampling1D(size=1)(blocks)
blocks = keras.layers.Conv1D(filters=2, kernel_size=21, padding = "same")(blocks)
blocks = keras.layers.normalization.BatchNormalization()(blocks)
blocks = keras.layers.Activation("selu")(blocks)

blocks = keras.layers.Conv1D(filters=1, kernel_size=1, padding = "same")(blocks)
blocks = keras.layers.Activation('linear')(blocks)

output_layer = blocks
autoencoder = keras.models.Model(input_layer, output_layer, name = "Autoencoder")

model_input = input_layer
output_autoencoder = autoencoder(model_input)

inputs = model_input
outputs = output_autoencoder


def my_vae_loss(y_true, y_pred):
        xent_loss = 561 * keras.losses.mean_squared_error(K.flatten(inputs), K.flatten(outputs))
        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        vae_loss = K.mean(xent_loss + kl_loss)
        #vae_loss = kl_loss
        return vae_loss
    
autoencoder.compile(optimizer='adam', loss=my_vae_loss, metrics = ["mse"])
#print(autoencoder.summary())



random_state = 0
dataset_name = "HARDataset"
dataset_path = "./datasets/HARDataset/"

X = pd.read_csv(dataset_path + "/train/" + "X_train.txt", header=None, delim_whitespace=True)
X = pd.DataFrame(X.values)
y = pd.read_csv(dataset_path + "/train/" + "y_train.txt", header = None, delim_whitespace=True).values
y_all = np.ravel(y).astype("int")
le = LabelEncoder()
le.fit(y_all)
y_all = le.transform(y_all)
X_all = X.values.reshape((X.shape[0], X.shape[1], 1))


    
X_test = pd.read_csv(dataset_path + "/test/" + "X_test.txt", header=None, delim_whitespace=True)
X_test = pd.DataFrame(X_test.values)
y_test = pd.read_csv(dataset_path + "/test/" + "y_test.txt", header = None, delim_whitespace=True).values
y_test = np.ravel(y_test).astype("int")
y_test = le.transform(y_test)
X_test = X_test.values.reshape((X_test.shape[0], X_test.shape[1], 1))

# BLACKBOX/EXPLANATION SETS SPLIT
X_train, X_exp, y_train, y_exp = train_test_split(X_all, y_all, 
                                                  test_size=0.3, stratify = y_all, random_state=random_state)

# BLACKBOX TRAIN/VALIDATION SETS SPLIT
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, 
                                                  test_size=0.2, stratify = y_train, random_state=random_state)

# EXPLANATION TRAIN/TEST SETS SPLIT
X_exp_train, X_exp_test, y_exp_train, y_exp_test = train_test_split(X_exp, y_exp, 
                                                                    test_size=0.2, 
                                                                    stratify = y_exp, 
                                                                    random_state=random_state)

# EXPLANATION TRAIN/VALIDATION SETS SPLIT
X_exp_train, X_exp_val, y_exp_train, y_exp_val = train_test_split(X_exp_train, y_exp_train, 
                                                                  test_size=0.2, 
                                                                  stratify = y_exp_train, 
                                                                  random_state=random_state)
n_timesteps, n_outputs, n_features = X_train.shape[1], len(np.unique(y_all)), 1 



reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=200, min_lr=0.0001)

"""
autoencoder.fit(X_exp_train, X_exp_train, epochs=1000, validation_data=(X_exp_val, X_exp_val), verbose = 1, 
                callbacks = [reduce_lr])
"""

autoencoder.load_weights("./autoencoder_checkpoints/HARDataset_debug_autoencoder_20191201_175329_best_weights_+17.473261_.hdf5")
print(autoencoder.evaluate(X_exp_train, X_exp_train))

from autoencoders import Autoencoder

params = {'input_shape': (561, 1), 
          'n_blocks': 8, 
          'latent_dim': 50, 
          'encoder_latent_layer_type': 'variational', 
          'encoder_args': {'filters': [2, 4, 8, 16, 32, 64, 128, 256], 
                           'kernel_size': [21, 18, 15, 13, 11, 8, 5, 3], 
                           'padding': 'same', 
                           'activation': 'selu', 
                           'pooling': [1, 1, 1, 1, 1, 1, 1, 1]}}

aut = Autoencoder(verbose = False, **params)
_, _, autoencoder = aut.build()
autoencoder.load_weights("./autoencoder_checkpoints/HARDataset_autoencoder_20191201_153104_best_weights_+17.406494_.hdf5")
print(autoencoder.evaluate(X_exp_train, X_exp_train))

# results are very similar