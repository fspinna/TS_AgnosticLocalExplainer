#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 21:35:33 2019

@author: francesco
"""

import keras

def build_lstm_autoencoder(n_timesteps, latent_dim):
    model = keras.models.Sequential()
    model.add(keras.layers.LSTM(latent_dim, input_shape=(n_timesteps,1)))
    model.add(keras.layers.advanced_activations.LeakyReLU(alpha=0.1))
    model.add(keras.layers.RepeatVector(n_timesteps))
    model.add(keras.layers.LSTM(latent_dim, return_sequences=True))
    model.add(keras.layers.advanced_activations.LeakyReLU(alpha=0.1))
    model.add(keras.layers.TimeDistributed(keras.layers.Dense(1)))
    model.compile(optimizer=keras.optimizers.Adam(), loss='mse')
    return model