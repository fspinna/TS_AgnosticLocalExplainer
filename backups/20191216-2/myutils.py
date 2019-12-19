#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 17:15:54 2019

@author: francesco
"""
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from agnosticlocalexplainer import AgnosticLocalExplainer

def reconstruction_blackbox_consistency(autoencoder, blackbox, dataset, keras = True, discriminative = False):
    if keras:
        if discriminative:
            decoded_tss = autoencoder.predict(dataset)[0]
        else:
            decoded_tss = autoencoder.predict(dataset)
        y_exp_pred = blackbox.predict(dataset)
        y_exp_recon_pred = blackbox.predict(decoded_tss)
        return accuracy_score(np.argmax(y_exp_pred, axis = 1), np.argmax(y_exp_recon_pred, axis = 1))
    else:
        # for KNN
        if discriminative:
            decoded_tss = autoencoder.predict(dataset)[0]
        else:
            decoded_tss = autoencoder.predict(dataset)
        y_exp_pred = blackbox.predict(dataset.reshape(dataset.shape[:2]))
        y_exp_recon_pred = blackbox.predict(decoded_tss.reshape(decoded_tss.shape[:2]))
        return accuracy_score(y_exp_pred, y_exp_recon_pred)
    
    
def df_to_sktime(df):
    # 2d dataframe to 2d sktime dataframe
    df_dict = {"dim_0": []}
    for series in df:
        df_dict["dim_0"].append(pd.Series(series))
    return pd.DataFrame(df_dict)
    
class BlackboxPredictWrapper(object):
    def __init__(self, blackbox, input_dimensions):
        self.blackbox = blackbox
        self.input_dimensions = input_dimensions
        
    def predict(self, dataset):
        # 3d dataset (batch, timesteps, 1)
        
        if self.input_dimensions == 2:
            dataset = dataset[:,:,0] # 3d to 2d array (batch, timesteps)

        prediction = self.blackbox.predict(dataset)
    
        if len(prediction.shape) > 1 and (prediction.shape[1] != 1):
            prediction = np.argmax(prediction, axis = 1) # from probability to  predicted class
            
        prediction = prediction.ravel() 
    
        return prediction
    
    def predict_proba(self, dataset):
        # X: 3d array (batch, timesteps, 1)
        if self.input_dimensions == 2:
            dataset = dataset[:,:,0] # 3d to 2d array (batch, timesteps)
            prediction = self.blackbox.predict_proba(dataset)
        else: prediction = self.blackbox.predict(dataset)
        return prediction
    



        
    
