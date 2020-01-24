#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 23:39:47 2019

@author: francesco
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from joblib import load
from blackboxes import build_resnet, build_simple_CNN
from myutils import BlackboxPredictWrapper
from autoencoders import Autoencoder, DiscriminativeAutoencoder
from sklearn.metrics import mean_squared_error, accuracy_score
from toy_autoencoders import build_lstm_autoencoder
import time
from sklearn.metrics import f1_score

import warnings
warnings.filterwarnings("ignore")

def benchmark_cbf(save_output = False):
    from pyts.datasets import make_cylinder_bell_funnel
    random_state = 0
    dataset_name = "cbf"
    print(dataset_name)
    X_all, y_all = make_cylinder_bell_funnel(n_samples = 600, random_state = random_state)
    X_all = X_all.reshape((X_all.shape[0], X_all.shape[1], 1))
    print("X SHAPE: ", X_all.shape)
    print("y SHAPE: ", y_all.shape)
    unique, counts = np.unique(y_all, return_counts=True)
    print("\nCLASSES BALANCE")
    for i, label in enumerate(unique):
        print(label, ": ", round(counts[i]/sum(counts), 2))
    # BLACKBOX/EXPLANATION SETS SPLIT
    X_train, X_exp, y_train, y_exp = train_test_split(X_all, y_all, 
                                                      test_size=0.3, stratify = y_all, random_state=random_state)
    
    # BLACKBOX TRAIN/TEST SETS SPLIT
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, 
                                                      test_size=0.2, stratify = y_train, random_state=random_state)
    
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
    
    print("SHAPES:")
    print("BLACKBOX TRAINING SET: ", X_train.shape)
    print("BLACKBOX VALIDATION SET: ", X_val.shape)
    print("BLACKBOX TEST SET: ", X_test.shape)
    print("EXPLANATION TRAINING SET: ", X_exp_train.shape)
    print("EXPLANATION VALIDATION SET: ", X_exp_val.shape)
    print("EXPLANATION TEST SET: ", X_exp_test.shape)
    
    n_timesteps, n_outputs, n_features = X_train.shape[1], len(np.unique(y_all)), 1 
    print("TIMESTEPS: ", n_timesteps)
    print("N. LABELS: ", n_outputs)
    
    
    
    results_blackboxes = {"resnet":[],
               "simplecnn":[],
               "knn": [],
              }
    
    results_blackboxes_rows = ["train_mse", 
                    "train_accuracy",
                    "train_f1_score",
                    "validation_mse", 
                    "validation_accuracy",
                    "validation_f1_score",
                    "test_mse", 
                    "test_accuracy",
                    "test_f1_score",
                   ]
    dataset_list = [(X_train, y_train), (X_val, y_val), (X_test, y_test)]
    dataset_list_exp = [(X_exp_train, y_exp_train), (X_exp_val, y_exp_val), (X_exp_test, y_exp_test)]
    
    resnet = build_resnet(n_timesteps, n_outputs)
    resnet.load_weights("./final_models/cbf/cbf_blackbox_resnet_20191106_145242_best_weights_+1.00_.hdf5")
    resnet_predict = BlackboxPredictWrapper(resnet, 3)
    
    simplecnn = build_simple_CNN(n_timesteps, n_outputs)
    simplecnn.load_weights("./final_models/cbf/cbf_blackbox_simpleCNN_20191106_145515_best_weights_+1.00_.hdf5")
    simplecnn_predict = BlackboxPredictWrapper(simplecnn, 3)
    
    knn = load("./final_models/cbf/cbf_blackbox_knn_20191106_145654.joblib")
    knn_predict = BlackboxPredictWrapper(knn, 2)
    
    predicts = [(resnet_predict, "resnet"), (simplecnn_predict, "simplecnn"), (knn_predict, "knn")]
    blackboxes = [(resnet, "resnet"), (simplecnn, "simplecnn"), (knn, "knn")]
    
    for i, blackbox_predict in enumerate(blackboxes):
        for dataset in dataset_list:
            real = dataset[1]
            predicted = predicts[i][0].predict(dataset[0])
            if blackbox_predict[1] in ["resnet", "simplecnn"]:
                prediction = blackbox_predict[0].evaluate(dataset[0], real)
                mse = prediction[0]
                accuracy = prediction[1]
            else:
                accuracy = blackbox_predict[0].score(dataset[0].reshape(dataset[0].shape[:2]), real)
                mse = mean_squared_error(real, blackbox_predict[0].predict(dataset[0].reshape(dataset[0].shape[:2])))
            #print(prediction.shape)
            f1 = f1_score(real, predicted, average = "weighted")
            results_blackboxes[blackbox_predict[1]].append(mse)
            results_blackboxes[blackbox_predict[1]].append(accuracy)
            results_blackboxes[blackbox_predict[1]].append(f1)
            
    results_blackboxes_df = pd.DataFrame(results_blackboxes, index = results_blackboxes_rows)  

    latent_dim = 2
    results_autoencoders = {"ae_cnn": [latent_dim],
               "vae_cnn": [latent_dim],
               "aae_cnn":[latent_dim],
               "avae_cnn": [latent_dim],
               "ae_lstm": [latent_dim]
              }
    
    results_autoencoders_rows = ["latent_dimension", "train_mse", "validation_mse", "test_mse"]
    
    
    for blackbox_predict in predicts:
        for X in ["train", "validation", "test"]:
            key = "reconstruction_" + blackbox_predict[1] + "_" + X + "_accuracy"
            results_autoencoders_rows.append(key)
    
    # STANDARD AUTOENCODER
    params = {'input_shape': (128, 1), 
              'n_blocks': 8, 
              'latent_dim': 2, 
              'encoder_latent_layer_type': 'dense', 
              'encoder_args': {'filters': [2, 4, 8, 16, 32, 64, 128, 256], 
                               'kernel_size': [21, 18, 15, 13, 11, 8, 5, 3], 
                               'padding': 'same', 
                               'activation': 'elu', 
                               'pooling': [1, 1, 1, 1, 1, 1, 1, 1]}}
    aut = Autoencoder(verbose = False, **params)
    _, _, ae_cnn = aut.build()
    ae_cnn.load_weights("./final_models/cbf/cbf_autoencoder_20191106_144056_best_weights_+1.0504_.hdf5")
    
    
    #VARIATIONAL AUTOENCODER
    params = {'input_shape': (128, 1), 
              'n_blocks': 8, 
              'latent_dim': 2, 
              'encoder_latent_layer_type': 'variational', 
              'encoder_args': {'filters': [2, 4, 8, 16, 32, 64, 128, 256], 
                               'kernel_size': [21, 18, 15, 13, 11, 8, 5, 3], 
                               'padding': 'same', 
                               'activation': 'elu', 
                               'pooling': [1, 1, 1, 1, 1, 1, 1, 1]}}
    aut = Autoencoder(verbose = False, **params)
    _, _, vae_cnn = aut.build()
    vae_cnn.load_weights("./final_models/cbf/cbf_autoencoder_20191106_144909_best_weights_+136.8745_.hdf5")
    
    
    #DISCRIMINATIVE AUTOENCODER
    params = {'input_shape': (128, 1), 
              'n_blocks': 8, 
              'latent_dim': 2, 
              'encoder_latent_layer_type': 'dense', 
              'encoder_args': {'filters': [2, 4, 8, 16, 32, 64, 128, 256], 
                               'kernel_size': [21, 18, 15, 13, 11, 8, 5, 3], 
                               'padding': 'same', 
                               'activation': 'elu', 
                               'pooling': [1, 1, 1, 1, 1, 1, 1, 1]}, 
              'discriminator_args': {'units': [100, 100], 
                                     'activation': 'relu'}, 
              'n_blocks_discriminator': 2}
    aut = DiscriminativeAutoencoder(verbose = False, **params)
    _, _, _, aae_cnn = aut.build()
    aae_cnn.load_weights("./final_models/cbf/cbf_autoencoder_20191106_150722_best_weights_+1.239848_.hdf5")
     
    #VARIATIONAL DISCRIMINATIVE AUTOENCODER
    params = {'input_shape': (128, 1), 
              'n_blocks': 8, 
              'latent_dim': 2, 
              'encoder_latent_layer_type': 'variational', 
              'encoder_args': {'filters': [2, 4, 8, 16, 32, 64, 128, 256], 
                               'kernel_size': [21, 18, 15, 13, 11, 8, 5, 3], 
                               'padding': 'same', 
                               'activation': 'elu', 
                               'pooling': [1, 1, 1, 1, 1, 1, 1, 1]}, 
              'discriminator_args': {'units': [100, 100], 
                                     'activation': 'relu'}, 
              'n_blocks_discriminator': 2}
    aut = DiscriminativeAutoencoder(verbose = False, **params)
    _, _, _, avae_cnn = aut.build()
    avae_cnn.load_weights("./final_models/cbf/cbf_autoencoder_20191106_153613_best_weights_+1.179660_.hdf5")
    
    # STANDARD (LSTM) AUTOENCODER
    ae_lstm = build_lstm_autoencoder(n_timesteps, 2)
    ae_lstm.load_weights("./final_models/cbf/cbf_lstm_autoencoder_20191130_211845_best_weights_+6.900020_.hdf5")
    
    autoencoders = [(ae_cnn, "ae_cnn"),(vae_cnn, "vae_cnn"),(aae_cnn, "aae_cnn"),(avae_cnn, "avae_cnn"), (ae_lstm, "ae_lstm")]
    
    
    for autoencoder in autoencoders:
        for dataset in dataset_list_exp:
            real = dataset[0]
            if ("aae" in autoencoder[1]) or ("avae" in autoencoder[1]):
                predicted = autoencoder[0].predict(real)[0]
            else:
                predicted = autoencoder[0].predict(real)
            mse = mean_squared_error(real.flatten(), predicted.flatten())
            results_autoencoders[autoencoder[1]].append(mse)
            
    for autoencoder in autoencoders:
        for blackbox_predict in predicts:
            for dataset in dataset_list_exp:
                real = dataset[0]
                real_class = blackbox_predict[0].predict(real)
                if ("aae" in autoencoder[1]) or ("avae" in autoencoder[1]):
                    predicted = autoencoder[0].predict(real)[0]
                else:
                    predicted = autoencoder[0].predict(real)
                predicted_class = blackbox_predict[0].predict(predicted)
                accuracy = accuracy_score(real_class, predicted_class)
                results_autoencoders[autoencoder[1]].append(accuracy)
                
        
    results_autoencoders_df = pd.DataFrame(results_autoencoders, index = results_autoencoders_rows)
    
    if save_output:
        results_autoencoders_df.to_csv("./final_models/" + dataset_name + "_autoencoders_" + time.strftime("%Y%m%d_%H%M%S") + ".csv", sep = ";")
        results_blackboxes_df.to_csv("./final_models/" + dataset_name + "_blackboxes_" + time.strftime("%Y%m%d_%H%M%S") + ".csv", sep = ";")
    
    return results_blackboxes_df, results_autoencoders_df
    


def benchmark_HAR(save_output = False):
    random_state = 0
    dataset_name = "HARDataset"
    print(dataset_name)
    dataset_path = "./datasets/HARDataset/"
    
    X = pd.read_csv(dataset_path + "/train/" + "X_train.txt", header=None, delim_whitespace=True)
    X = pd.DataFrame(X.values)
    y = pd.read_csv(dataset_path + "/train/" + "y_train.txt", header = None, delim_whitespace=True).values
    y_all = np.ravel(y).astype("int")
    le = LabelEncoder()
    le.fit(y_all)
    y_all = le.transform(y_all)
    X_all = X.values.reshape((X.shape[0], X.shape[1], 1))
    
    print("X_train SHAPE: ", X_all.shape)
    print("y_train SHAPE: ", y_all.shape)
    unique, counts = np.unique(y_all, return_counts=True)
    print("\nCLASSES BALANCE")
    for i, label in enumerate(unique):
        print(label, ": ", round(counts[i]/sum(counts), 2))
        
    X_test = pd.read_csv(dataset_path + "/test/" + "X_test.txt", header=None, delim_whitespace=True)
    X_test = pd.DataFrame(X_test.values)
    y_test = pd.read_csv(dataset_path + "/test/" + "y_test.txt", header = None, delim_whitespace=True).values
    y_test = np.ravel(y_test).astype("int")
    y_test = le.transform(y_test)
    X_test = X_test.values.reshape((X_test.shape[0], X_test.shape[1], 1))
    print("X_test SHAPE: ", X_test.shape)
    print("y_test SHAPE: ", y_test.shape)
    
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
    
    print("SHAPES:")
    print("BLACKBOX TRAINING SET: ", X_train.shape)
    print("BLACKBOX VALIDATION SET: ", X_val.shape)
    print("BLACKBOX TEST SET: ", X_test.shape)
    print("EXPLANATION TRAINING SET: ", X_exp_train.shape)
    print("EXPLANATION VALIDATION SET: ", X_exp_val.shape)
    print("EXPLANATION TEST SET: ", X_exp_test.shape)
    
    n_timesteps, n_outputs, n_features = X_train.shape[1], len(np.unique(y_all)), 1 
    print("TIMESTEPS: ", n_timesteps)
    print("N. LABELS: ", n_outputs)
    
    
    
    results_blackboxes = {"resnet":[],
               "simplecnn":[],
               "knn": [],
              }
    
    results_blackboxes_rows = ["train_mse", 
                    "train_accuracy",
                    "train_f1_score",
                    "validation_mse", 
                    "validation_accuracy",
                    "validation_f1_score",
                    "test_mse", 
                    "test_accuracy",
                    "test_f1_score",
                   ]
    dataset_list = [(X_train, y_train), (X_val, y_val), (X_test, y_test)]
    dataset_list_exp = [(X_exp_train, y_exp_train), (X_exp_val, y_exp_val), (X_exp_test, y_exp_test)]
    
    resnet = build_resnet(n_timesteps, n_outputs)
    resnet.load_weights("./final_models/HARDataset/HARDataset_blackbox_resnet_20191028_172136_best_weights_+0.99_.hdf5")
    resnet_predict = BlackboxPredictWrapper(resnet, 3)
    
    simplecnn = build_simple_CNN(n_timesteps, n_outputs)
    simplecnn.load_weights("./final_models/HARDataset/HARDataset_blackbox_simpleCNN_20191029_153407_best_weights_+0.94_.hdf5")
    simplecnn_predict = BlackboxPredictWrapper(simplecnn, 3)
    
    knn = load("./final_models/HARDataset/HARDataset_blackbox_knn_20191031_111540.joblib")
    knn_predict = BlackboxPredictWrapper(knn, 2)
    
    predicts = [(resnet_predict, "resnet"), (simplecnn_predict, "simplecnn"), (knn_predict, "knn")]
    blackboxes = [(resnet, "resnet"), (simplecnn, "simplecnn"), (knn, "knn")]
    
    for i, blackbox_predict in enumerate(blackboxes):
        for dataset in dataset_list:
            real = dataset[1]
            predicted = predicts[i][0].predict(dataset[0])
            if blackbox_predict[1] in ["resnet", "simplecnn"]:
                prediction = blackbox_predict[0].evaluate(dataset[0], real)
                mse = prediction[0]
                accuracy = prediction[1]
            else:
                accuracy = blackbox_predict[0].score(dataset[0].reshape(dataset[0].shape[:2]), real)
                mse = mean_squared_error(real, blackbox_predict[0].predict(dataset[0].reshape(dataset[0].shape[:2])))
            #print(prediction.shape)
            f1 = f1_score(real, predicted, average = "weighted")
            results_blackboxes[blackbox_predict[1]].append(mse)
            results_blackboxes[blackbox_predict[1]].append(accuracy)
            results_blackboxes[blackbox_predict[1]].append(f1)
            
    results_blackboxes_df = pd.DataFrame(results_blackboxes, index = results_blackboxes_rows)  

    latent_dim = 50
    results_autoencoders = {"ae_cnn": [latent_dim],
                            "vae_cnn": [latent_dim],
                            "aae_cnn": [latent_dim],
               "ae_lstm": [latent_dim]
              }
    
    results_autoencoders_rows = ["latent_dimension", "train_mse", "validation_mse", "test_mse"]
    
    
    for blackbox_predict in predicts:
        for X in ["train", "validation", "test"]:
            key = "reconstruction_" + blackbox_predict[1] + "_" + X + "_accuracy"
            results_autoencoders_rows.append(key)
    
    # STANDARD AUTOENCODER
    params = {"input_shape": (n_timesteps,1),
              "n_blocks": 8, 
              "latent_dim": 50,
              "encoder_latent_layer_type": "dense",
              "encoder_args": {"filters":[2,4,8,16,32,64,128,256], 
                                "kernel_size":[21,18,15,13,11,8,5,3], 
                                "padding":"same", 
                                "activation":"selu", 
                                "pooling":[1,1,1,1,1,1,1,1]}
             }
    aut = Autoencoder(verbose = False, **params)
    _, _, ae_cnn = aut.build()
    ae_cnn.load_weights("./final_models/HARDataset/HARDataset_autoencoder_20191031_212226_best_weights_+0.008519_.hdf5")
    
    # VARIATIONAL AUTOENCODER
    params = {"input_shape": (n_timesteps,1),
              "n_blocks": 8, 
              "latent_dim": 50,
              "encoder_latent_layer_type": "variational",
              "encoder_args": {"filters":[2,4,8,16,32,64,128,256], 
                                "kernel_size":[21,18,15,13,11,8,5,3], 
                                "padding":"same", 
                                "activation":"selu", 
                                "pooling":[1,1,1,1,1,1,1,1]}
             }
    aut = Autoencoder(verbose = False, **params)
    _, _, vae_cnn = aut.build()
    vae_cnn.load_weights("./final_models/HARDataset/HARDataset_autoencoder_20191201_153104_best_weights_+17.406494_.hdf5")
    
    #DISCRIMINATIVE AUTOENCODER
    params = {"input_shape": (561,1),
              "n_blocks": 8, 
              "latent_dim": 50,
              "encoder_latent_layer_type": "dense",
              "encoder_args": {"filters":[2,4,8,16,32,64,128,256], 
                                "kernel_size":[21,18,15,13,11,8,5,3], 
                                "padding":"same", 
                                "activation":"selu", 
                                "pooling":[1,1,1,1,1,1,1,1]},
              "discriminator_args": {"units": [100,100],
                                     "activation": "relu"},
              "n_blocks_discriminator": 2}
    aut = DiscriminativeAutoencoder(verbose = False, **params)
    _, _, _, aae_cnn = aut.build()
    aae_cnn.load_weights("./final_models/HARDataset/HARDataset_autoencoder_20191202_best_weights_+0.007953_.hdf5")
    
    # STANDARD (LSTM) AUTOENCODER
    ae_lstm = build_lstm_autoencoder(n_timesteps, latent_dim)
    ae_lstm.load_weights("./final_models/HARDataset/HARDataset_lstm_autoencoder_20191130_093909_best_weights_+0.070170_.hdf5")
    
    autoencoders = [(ae_cnn, "ae_cnn"),(vae_cnn, "vae_cnn"),(aae_cnn, "aae_cnn"),(ae_lstm, "ae_lstm")]
    
    
    for autoencoder in autoencoders:
        for dataset in dataset_list_exp:
            real = dataset[0]
            if ("aae" in autoencoder[1]) or ("avae" in autoencoder[1]):
                predicted = autoencoder[0].predict(real)[0]
            else:
                predicted = autoencoder[0].predict(real)
            mse = mean_squared_error(real.flatten(), predicted.flatten())
            results_autoencoders[autoencoder[1]].append(mse)
            
    for autoencoder in autoencoders:
        for blackbox_predict in predicts:
            for dataset in dataset_list_exp:
                real = dataset[0]
                real_class = blackbox_predict[0].predict(real)
                if ("aae" in autoencoder[1]) or ("avae" in autoencoder[1]):
                    predicted = autoencoder[0].predict(real)[0]
                else:
                    predicted = autoencoder[0].predict(real)
                predicted_class = blackbox_predict[0].predict(predicted)
                accuracy = accuracy_score(real_class, predicted_class)
                results_autoencoders[autoencoder[1]].append(accuracy)
                
        
    results_autoencoders_df = pd.DataFrame(results_autoencoders, index = results_autoencoders_rows)
    
    if save_output:
        results_autoencoders_df.to_csv("./final_models/" + dataset_name + "_autoencoders_" + time.strftime("%Y%m%d_%H%M%S") + ".csv", sep = ";")
        results_blackboxes_df.to_csv("./final_models/" + dataset_name + "_blackboxes_" + time.strftime("%Y%m%d_%H%M%S") + ".csv", sep = ";")
    
    return results_blackboxes_df, results_autoencoders_df


def benchmark_phalanges(save_output = False):
    random_state = 0
    dataset_path = "./datasets/PhalangesOutlinesCorrect/"
    dataset_name = "phalanges"
    print(dataset_name)
    X = pd.read_csv(dataset_path + "PhalangesOutlinesCorrect_TRAIN.txt", header=None, delim_whitespace=True)
    y_all = np.array(X[0]).astype("int")
    X_all = X.iloc[:,1:].values
    X_all = X_all.reshape((X_all.shape[0], X_all.shape[1], 1))
    print("X SHAPE: ", X_all.shape)
    print("y SHAPE: ", y_all.shape)
    unique, counts = np.unique(y_all, return_counts=True)
    print("\nCLASSES BALANCE")
    for i, label in enumerate(unique):
        print(label, ": ", round(counts[i]/sum(counts), 2))
    X_test = pd.read_csv(dataset_path + "PhalangesOutlinesCorrect_TEST.txt", header=None, delim_whitespace=True)
    y_test = np.array(X_test[0]).astype("int")
    X_test = X_test.iloc[:,1:].values
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    print("X SHAPE: ", X_test.shape)
    print("y SHAPE: ", y_test.shape)
    
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
    
    print("SHAPES:")
    print("BLACKBOX TRAINING SET: ", X_train.shape)
    print("BLACKBOX VALIDATION SET: ", X_val.shape)
    print("BLACKBOX TEST SET: ", X_test.shape)
    print("EXPLANATION TRAINING SET: ", X_exp_train.shape)
    print("EXPLANATION VALIDATION SET: ", X_exp_val.shape)
    print("EXPLANATION TEST SET: ", X_exp_test.shape)
    n_timesteps, n_outputs, n_features = X_train.shape[1], len(np.unique(y_all)), 1 
    print("TIMESTEPS: ", n_timesteps)
    print("N. LABELS: ", n_outputs)
    
    
    
    results_blackboxes = {"resnet":[],
               "simplecnn":[],
               "knn": [],
              }
    
    results_blackboxes_rows = ["train_mse", 
                    "train_accuracy",
                    "train_f1_score",
                    "validation_mse", 
                    "validation_accuracy",
                    "validation_f1_score",
                    "test_mse", 
                    "test_accuracy",
                    "test_f1_score",
                   ]
    dataset_list = [(X_train, y_train), (X_val, y_val), (X_test, y_test)]
    dataset_list_exp = [(X_exp_train, y_exp_train), (X_exp_val, y_exp_val), (X_exp_test, y_exp_test)]
    
    resnet = build_resnet(n_timesteps, n_outputs)
    resnet.load_weights("./final_models/phalanges/phalanges_blackbox_resnet_20191101_164247_best_weights_+0.86_.hdf5")
    resnet_predict = BlackboxPredictWrapper(resnet, 3)
    
    simplecnn = build_simple_CNN(n_timesteps, n_outputs)
    simplecnn.load_weights("./final_models/phalanges/phalanges_blackbox_simpleCNN_20191101_170209_best_weights_+0.83_.hdf5")
    simplecnn_predict = BlackboxPredictWrapper(simplecnn, 3)
    
    knn = load("./final_models/phalanges/phalanges_blackbox_knn_20191101_171534.joblib")
    knn_predict = BlackboxPredictWrapper(knn, 2)
    
    predicts = [(resnet_predict, "resnet"), (simplecnn_predict, "simplecnn"), (knn_predict, "knn")]
    blackboxes = [(resnet, "resnet"), (simplecnn, "simplecnn"), (knn, "knn")]
    
    for i, blackbox_predict in enumerate(blackboxes):
        for dataset in dataset_list:
            real = dataset[1]
            predicted = predicts[i][0].predict(dataset[0])
            if blackbox_predict[1] in ["resnet", "simplecnn"]:
                prediction = blackbox_predict[0].evaluate(dataset[0], real)
                mse = prediction[0]
                accuracy = prediction[1]
            else:
                accuracy = blackbox_predict[0].score(dataset[0].reshape(dataset[0].shape[:2]), real)
                mse = mean_squared_error(real, blackbox_predict[0].predict(dataset[0].reshape(dataset[0].shape[:2])))
            #print(prediction.shape)
            f1 = f1_score(real, predicted, average = "weighted")
            results_blackboxes[blackbox_predict[1]].append(mse)
            results_blackboxes[blackbox_predict[1]].append(accuracy)
            results_blackboxes[blackbox_predict[1]].append(f1)
            
    results_blackboxes_df = pd.DataFrame(results_blackboxes, index = results_blackboxes_rows)  

    latent_dim = 40
    results_autoencoders = {"ae_cnn": [latent_dim],
                            "vae_cnn": [latent_dim],
                            "aae_cnn": [latent_dim],
               "ae_lstm": [latent_dim]
              }
    
    results_autoencoders_rows = ["latent_dimension", "train_mse", "validation_mse", "test_mse"]
    
    
    for blackbox_predict in predicts:
        for X in ["train", "validation", "test"]:
            key = "reconstruction_" + blackbox_predict[1] + "_" + X + "_accuracy"
            results_autoencoders_rows.append(key)
    
    # STANDARD AUTOENCODER
    params = {"input_shape": (n_timesteps,1),
              "n_blocks": 8, 
              "latent_dim": 40,
              "encoder_latent_layer_type": "simple",
              "encoder_args": {"filters":[2,4,8,16,32,64,128,256], 
                                "kernel_size":[21,18,15,13,11,8,5,3], 
                                "padding":"same", 
                                "activation":"elu", 
                                "pooling":[1,1,1,1,1,1,1,2]}
             }
    aut = Autoencoder(verbose = False, **params)
    _, _, ae_cnn = aut.build()
    ae_cnn.load_weights("./final_models/phalanges/phalanges_autoencoder_20191103_211535_best_weights_+0.0010_.hdf5")
    
    # VARIATIONAL AUTOENCODER
    params = {"input_shape": (n_timesteps,1),
              "n_blocks": 8, 
              "latent_dim": 40,
              "encoder_latent_layer_type": "variational",
              "encoder_args": {"filters":[2,4,8,16,32,64,128,256], 
                                "kernel_size":[21,18,15,13,11,8,5,3], 
                                "padding":"same", 
                                "activation":"elu", 
                                "pooling":[1,1,1,1,1,1,1,2]}
             }
    aut = Autoencoder(verbose = False, **params)
    _, _, vae_cnn = aut.build()
    vae_cnn.load_weights("./final_models/phalanges/phalanges_autoencoder_20191201_145321_best_weights_+2.9536_.hdf5")
    
    #DISCRIMINATIVE AUTOENCODER
    params = {'input_shape': (80, 1), 
              'n_blocks': 8, 
              'latent_dim': 40, 
              'encoder_latent_layer_type': 'simple', 
              'encoder_args': {'filters': [2, 4, 8, 16, 32, 64, 128, 256], 
                               'kernel_size': [21, 18, 15, 13, 11, 8, 5, 3], 
                               'padding': 'same', 
                               'activation': 'elu', 
                               'pooling': [1, 1, 1, 1, 1, 1, 1, 2]}, 
                               'discriminator_args': {'units': [100, 100], 'activation': 'relu'}, 
                               'n_blocks_discriminator': 2}
    aut = DiscriminativeAutoencoder(verbose = False, **params)
    _, _, _, aae_cnn = aut.build()
    aae_cnn.load_weights("./final_models/phalanges/phalanges_autoencoder_20191202_133021_best_weights_+0.001379_.hdf5")
    
    
    # STANDARD (LSTM) AUTOENCODER
    ae_lstm = build_lstm_autoencoder(n_timesteps, latent_dim)
    ae_lstm.load_weights("./final_models/phalanges/phalanges_lstm_autoencoder_20191130_214525_best_weights_+0.014177_.hdf5")
    
    autoencoders = [(ae_cnn, "ae_cnn"),(vae_cnn, "vae_cnn"),(aae_cnn, "aae_cnn"),(ae_lstm, "ae_lstm")]
    
    
    for autoencoder in autoencoders:
        for dataset in dataset_list_exp:
            real = dataset[0]
            if ("aae" in autoencoder[1]) or ("avae" in autoencoder[1]):
                predicted = autoencoder[0].predict(real)[0]
            else:
                predicted = autoencoder[0].predict(real)
            mse = mean_squared_error(real.flatten(), predicted.flatten())
            results_autoencoders[autoencoder[1]].append(mse)
            
    for autoencoder in autoencoders:
        for blackbox_predict in predicts:
            for dataset in dataset_list_exp:
                real = dataset[0]
                real_class = blackbox_predict[0].predict(real)
                if ("aae" in autoencoder[1]) or ("avae" in autoencoder[1]):
                    predicted = autoencoder[0].predict(real)[0]
                else:
                    predicted = autoencoder[0].predict(real)
                predicted_class = blackbox_predict[0].predict(predicted)
                accuracy = accuracy_score(real_class, predicted_class)
                results_autoencoders[autoencoder[1]].append(accuracy)
                
        
    results_autoencoders_df = pd.DataFrame(results_autoencoders, index = results_autoencoders_rows)
    
    if save_output:
        results_autoencoders_df.to_csv("./final_models/" + dataset_name + "_autoencoders_" + time.strftime("%Y%m%d_%H%M%S") + ".csv", sep = ";")
        results_blackboxes_df.to_csv("./final_models/" + dataset_name + "_blackboxes_" + time.strftime("%Y%m%d_%H%M%S") + ".csv", sep = ";")
    
    return results_blackboxes_df, results_autoencoders_df


def benchmark_epileptic(save_output = False):
    random_state = 0
    dataset_path = "./datasets/EpilepticSeizureRecognition/"
    dataset_name = "EpilepticSeizureRecognition"
    print(dataset_name)
    X = pd.read_csv(dataset_path + "data.csv", index_col = 0)
    y = np.array(X["y"])
    y_all = np.ravel(y).astype("int")
    for i in range(2,6):
        y_all[y_all == i] = 2
    le = LabelEncoder()
    le.fit(y_all)
    y_all = le.transform(y_all)
    X_all = X.drop("y", axis = 1).values
    from imblearn.under_sampling import RandomUnderSampler # doctest: +NORMALIZE_WHITESPACE
    rus = RandomUnderSampler(random_state=random_state)
    X_all, y_all = rus.fit_resample(X_all, y_all)
    X_all = X_all.reshape((X_all.shape[0], X_all.shape[1], 1))
    
    print("X SHAPE: ", X_all.shape)
    print("y SHAPE: ", y_all.shape)
    unique, counts = np.unique(y_all, return_counts=True)
    print("\nCLASSES BALANCE")
    for i, label in enumerate(unique):
        print(label, ": ", round(counts[i]/sum(counts), 2))
        
    # BLACKBOX TRAIN/TEST SETS SPLIT
    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, 
                                                      test_size=0.2, stratify = y_all, random_state=random_state)
    
    # BLACKBOX/EXPLANATION SETS SPLIT
    X_train, X_exp, y_train, y_exp = train_test_split(X_train, y_train, 
                                                      test_size=0.3, stratify = y_train, random_state=random_state)
    
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
    
    print("SHAPES:")
    print("BLACKBOX TRAINING SET: ", X_train.shape)
    print("BLACKBOX VALIDATION SET: ", X_val.shape)
    print("BLACKBOX TEST SET: ", X_test.shape)
    print("EXPLANATION TRAINING SET: ", X_exp_train.shape)
    print("EXPLANATION VALIDATION SET: ", X_exp_val.shape)
    print("EXPLANATION TEST SET: ", X_exp_test.shape)
    
    n_timesteps, n_outputs, n_features = X_train.shape[1], len(np.unique(y_all)), 1 
    print("TIMESTEPS: ", n_timesteps)
    print("N. LABELS: ", n_outputs)
    
    
    results_blackboxes = {"resnet":[],
               "simplecnn":[],
               "knn": [],
              }
    
    results_blackboxes_rows = ["train_mse", 
                    "train_accuracy",
                    "train_f1_score",
                    "validation_mse", 
                    "validation_accuracy",
                    "validation_f1_score",
                    "test_mse", 
                    "test_accuracy",
                    "test_f1_score",
                   ]
    dataset_list = [(X_train, y_train), (X_val, y_val), (X_test, y_test)]
    dataset_list_exp = [(X_exp_train, y_exp_train), (X_exp_val, y_exp_val), (X_exp_test, y_exp_test)]
    
    resnet = build_resnet(n_timesteps, n_outputs)
    resnet.load_weights("./final_models/EpilepticSeizureRecognition/EpilepticSeizureRecognition_blackbox_resnet_20200105_233014_best_weights_+0.99_.hdf5")
    resnet_predict = BlackboxPredictWrapper(resnet, 3)
    
    simplecnn = build_simple_CNN(n_timesteps, n_outputs)
    simplecnn.load_weights("./final_models/EpilepticSeizureRecognition/EpilepticSeizureRecognition_blackbox_simpleCNN_20200105_225722_best_weights_+0.98_.hdf5")
    simplecnn_predict = BlackboxPredictWrapper(simplecnn, 3)
    
    knn = load("./final_models/EpilepticSeizureRecognition/EpilepticSeizureRecognition_blackbox_knn_20200105_225631.joblib")
    knn_predict = BlackboxPredictWrapper(knn, 2)
    
    predicts = [(resnet_predict, "resnet"), (simplecnn_predict, "simplecnn"), (knn_predict, "knn")]
    blackboxes = [(resnet, "resnet"), (simplecnn, "simplecnn"), (knn, "knn")]
    
    for i, blackbox_predict in enumerate(blackboxes):
        for dataset in dataset_list:
            real = dataset[1]
            predicted = predicts[i][0].predict(dataset[0])
            if blackbox_predict[1] in ["resnet", "simplecnn"]:
                prediction = blackbox_predict[0].evaluate(dataset[0], real)
                mse = prediction[0]
                accuracy = prediction[1]
            else:
                accuracy = blackbox_predict[0].score(dataset[0].reshape(dataset[0].shape[:2]), real)
                mse = mean_squared_error(real, blackbox_predict[0].predict(dataset[0].reshape(dataset[0].shape[:2])))
            #print(prediction.shape)
            f1 = f1_score(real, predicted, average = "weighted")
            results_blackboxes[blackbox_predict[1]].append(mse)
            results_blackboxes[blackbox_predict[1]].append(accuracy)
            results_blackboxes[blackbox_predict[1]].append(f1)
            
    results_blackboxes_df = pd.DataFrame(results_blackboxes, index = results_blackboxes_rows)  

    latent_dim = 30
    results_autoencoders = {"ae_cnn": [latent_dim],
                            "vae_cnn": [latent_dim],
                            "aae_cnn": [latent_dim],
               "ae_lstm": [latent_dim]
              }
    
    results_autoencoders_rows = ["latent_dimension", "train_mse", "validation_mse", "test_mse"]
    
    
    for blackbox_predict in predicts:
        for X in ["train", "validation", "test"]:
            key = "reconstruction_" + blackbox_predict[1] + "_" + X + "_accuracy"
            results_autoencoders_rows.append(key)
    
    # STANDARD AUTOENCODER
    params = {'input_shape': (178, 1), 
              'n_blocks': 8, 
              'latent_dim': 30, 
              'encoder_latent_layer_type': 'dense', 
              'encoder_args': {'filters': [2, 4, 8, 16, 32, 64, 128, 256], 
                               'kernel_size': [21, 18, 15, 13, 11, 8, 5, 3], 
                               'padding': 'same', 
                               'activation': 'elu', 
                               'pooling': [1, 1, 1, 1, 1, 1, 1, 1]}}
    aut = Autoencoder(verbose = False, **params)
    _, _, ae_cnn = aut.build()
    ae_cnn.load_weights("./final_models/EpilepticSeizureRecognition/EpilepticSeizureRecognition_autoencoder_20200106_111007_best_weights_+14872.8621_.hdf5")
    
    # VARIATIONAL AUTOENCODER
    params = {'input_shape': (178, 1), 
              'n_blocks': 8, 
              'latent_dim': 30, 
              'encoder_latent_layer_type': 'variational', 
              'encoder_args': {'filters': [2, 4, 8, 16, 32, 64, 128, 256], 
                               'kernel_size': [21, 18, 15, 13, 11, 8, 5, 3], 
                               'padding': 'same', 
                               'activation': 'elu', 
                               'pooling': [1, 1, 1, 1, 1, 1, 1, 1]}}
    aut = Autoencoder(verbose = False, **params)
    _, _, vae_cnn = aut.build()
    vae_cnn.load_weights("./final_models/EpilepticSeizureRecognition/EpilepticSeizureRecognition_autoencoder_20200106_115956_best_weights_+4548653.0325_.hdf5")
    
    #DISCRIMINATIVE AUTOENCODER
    params = {'input_shape': (178, 1), 
              'n_blocks': 8, 
              'latent_dim': 30, 
              'encoder_latent_layer_type': 'dense', 
              'encoder_args': {'filters': [2, 4, 8, 16, 32, 64, 128, 256], 
                               'kernel_size': [21, 18, 15, 13, 11, 8, 5, 3], 
                               'padding': 'same', 
                               'activation': 'elu', 
                               'pooling': [1, 1, 1, 1, 1, 1, 1, 1]}, 
                               'discriminator_args': {'units': [100, 100], 
                                                      'activation': 'relu'}, 
                                                      'n_blocks_discriminator': 2}
    aut = DiscriminativeAutoencoder(verbose = False, **params)
    _, _, _, aae_cnn = aut.build()
    aae_cnn.load_weights("./final_models/EpilepticSeizureRecognition/EpilepticSeizureRecognition_autoencoder_20200106_122042_best_weights_+19803.450249_.hdf5")
    
    
    # STANDARD (LSTM) AUTOENCODER
    ae_lstm = build_lstm_autoencoder(n_timesteps, latent_dim)
    ae_lstm.load_weights("./final_models/EpilepticSeizureRecognition/EpilepticSeizureRecognition_lstm_autoencoder_20200106_142105_best_weights_+59024.272731_.hdf5")
    
    autoencoders = [(ae_cnn, "ae_cnn"),(vae_cnn, "vae_cnn"),(aae_cnn, "aae_cnn"),(ae_lstm, "ae_lstm")]
    
    
    for autoencoder in autoencoders:
        for dataset in dataset_list_exp:
            real = dataset[0]
            if ("aae" in autoencoder[1]) or ("avae" in autoencoder[1]):
                predicted = autoencoder[0].predict(real)[0]
            else:
                predicted = autoencoder[0].predict(real)
            mse = mean_squared_error(real.flatten(), predicted.flatten())
            results_autoencoders[autoencoder[1]].append(mse)
            
    for autoencoder in autoencoders:
        for blackbox_predict in predicts:
            for dataset in dataset_list_exp:
                real = dataset[0]
                real_class = blackbox_predict[0].predict(real)
                if ("aae" in autoencoder[1]) or ("avae" in autoencoder[1]):
                    predicted = autoencoder[0].predict(real)[0]
                else:
                    predicted = autoencoder[0].predict(real)
                predicted_class = blackbox_predict[0].predict(predicted)
                accuracy = accuracy_score(real_class, predicted_class)
                results_autoencoders[autoencoder[1]].append(accuracy)
                
        
    results_autoencoders_df = pd.DataFrame(results_autoencoders, index = results_autoencoders_rows)
    
    if save_output:
        results_autoencoders_df.to_csv("./final_models/" + dataset_name + "_autoencoders_" + time.strftime("%Y%m%d_%H%M%S") + ".csv", sep = ";")
        results_blackboxes_df.to_csv("./final_models/" + dataset_name + "_blackboxes_" + time.strftime("%Y%m%d_%H%M%S") + ".csv", sep = ";")
    
    return results_blackboxes_df, results_autoencoders_df
    
    