#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 17:36:35 2019

@author: francesco
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from joblib import load
from blackboxes import build_resnet, build_simple_CNN
from myutils import reconstruction_blackbox_consistency
from autoencoders import Autoencoder, DiscriminativeAutoencoder
from sklearn.metrics import mean_squared_error
from toy_autoencoders import build_lstm_autoencoder
import time

import warnings
warnings.filterwarnings("ignore")

def benchmark_cbf():
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
    
    
    results = {"resnet":[],
               "simplecnn":[],
               "knn": [],
               "autoencoder": [],
               "vae": [],
               "discriminative_autoencoder":[],
               "discriminative_vae": []
              }
    results_rows = ["train_mse", 
                    "train_accuracy",
                    "validation_mse", 
                    "validation_accuracy",
                    "test_mse", 
                    "test_accuracy",
                    "latent_dimension",
                    "reconstruction_blackbox_accuracy_train_ae",
                    "reconstruction_blackbox_accuracy_validation_ae",
                    "reconstruction_blackbox_accuracy_test_ae",
                    "reconstruction_blackbox_accuracy_train_vae",
                    "reconstruction_blackbox_accuracy_validation_vae",
                    "reconstruction_blackbox_accuracy_test_vae",
                    "reconstruction_blackbox_accuracy_train_dae",
                    "reconstruction_blackbox_accuracy_validation_dae",
                    "reconstruction_blackbox_accuracy_test_dae",
                    "reconstruction_blackbox_accuracy_train_dvae",
                    "reconstruction_blackbox_accuracy_validation_dvae",
                    "reconstruction_blackbox_accuracy_test_dvae",
                   ]
    
    resnet = build_resnet(n_timesteps, n_outputs)
    resnet.load_weights("./final_models/cbf/cbf_blackbox_resnet_20191106_145242_best_weights_+1.00_.hdf5")
    results["resnet"].extend(resnet.evaluate(X_train, y_train))
    results["resnet"].extend(resnet.evaluate(X_val, y_val))
    results["resnet"].extend(resnet.evaluate(X_test, y_test))
    results["resnet"].append(np.NaN)
    
    
    simplecnn = build_simple_CNN(n_timesteps, n_outputs)
    simplecnn.load_weights("./final_models/cbf/cbf_blackbox_simpleCNN_20191106_145515_best_weights_+1.00_.hdf5")
    results["simplecnn"].extend(simplecnn.evaluate(X_train, y_train))
    results["simplecnn"].extend(simplecnn.evaluate(X_val, y_val))
    results["simplecnn"].extend(simplecnn.evaluate(X_test, y_test))
    results["simplecnn"].append(np.NaN)
    
    knn = load("./final_models/cbf/cbf_blackbox_knn_20191106_145654.joblib")
    results["knn"].append(mean_squared_error(y_train, knn.predict(X_train.reshape(X_train.shape[:2]))))
    results["knn"].append(knn.score(X_train.reshape(X_train.shape[:2]), y_train))
    results["knn"].append(mean_squared_error(y_val, knn.predict(X_val.reshape(X_val.shape[:2]))))
    results["knn"].append(knn.score(X_val.reshape(X_val.shape[:2]), y_val))
    results["knn"].append(mean_squared_error(y_test, knn.predict(X_test.reshape(X_test.shape[:2]))))
    results["knn"].append(knn.score(X_test.reshape(X_test.shape[:2]), y_test))
    results["knn"].append(np.NaN)
    
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
    _, _, autoencoder = aut.build()
    autoencoder.load_weights("./final_models/cbf/cbf_autoencoder_20191106_144056_best_weights_+1.0504_.hdf5")
    autoencoder_name = "autoencoder"
    results[autoencoder_name].append(autoencoder.evaluate(X_exp_train, X_exp_train))
    results[autoencoder_name].append(np.NaN)
    results[autoencoder_name].append(autoencoder.evaluate(X_exp_val, X_exp_val))
    results[autoencoder_name].append(np.NaN)
    results[autoencoder_name].append(autoencoder.evaluate(X_exp_test, X_exp_test))
    results[autoencoder_name].append(np.NaN)
    results[autoencoder_name].append(params["latent_dim"])
    for i in range(12):
        results[autoencoder_name].append(np.NaN)
    results["resnet"].append(reconstruction_blackbox_consistency(autoencoder, resnet, X_exp_train))
    results["resnet"].append(reconstruction_blackbox_consistency(autoencoder, resnet, X_exp_val))
    results["resnet"].append(reconstruction_blackbox_consistency(autoencoder, resnet, X_exp_test))
    results["simplecnn"].append(reconstruction_blackbox_consistency(autoencoder, simplecnn, X_exp_train))
    results["simplecnn"].append(reconstruction_blackbox_consistency(autoencoder, simplecnn, X_exp_val))
    results["simplecnn"].append(reconstruction_blackbox_consistency(autoencoder, simplecnn, X_exp_test))
    results["knn"].append(reconstruction_blackbox_consistency(autoencoder, knn, X_exp_train, keras = False))
    results["knn"].append(reconstruction_blackbox_consistency(autoencoder, knn, X_exp_val, keras = False))
    results["knn"].append(reconstruction_blackbox_consistency(autoencoder, knn, X_exp_test, keras = False))
    
    
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
    _, _, autoencoder = aut.build()
    autoencoder.load_weights("./final_models/cbf/cbf_autoencoder_20191106_144909_best_weights_+136.8745_.hdf5")
    autoencoder_name = "vae"
    results[autoencoder_name].append(autoencoder.evaluate(X_exp_train, X_exp_train)[1])
    results[autoencoder_name].append(np.NaN)
    results[autoencoder_name].append(autoencoder.evaluate(X_exp_val, X_exp_val)[1])
    results[autoencoder_name].append(np.NaN)
    results[autoencoder_name].append(autoencoder.evaluate(X_exp_test, X_exp_test)[1])
    results[autoencoder_name].append(np.NaN)
    results[autoencoder_name].append(params["latent_dim"])
    for i in range(12):
        results[autoencoder_name].append(np.NaN)
    results["resnet"].append(reconstruction_blackbox_consistency(autoencoder, resnet, X_exp_train))
    results["resnet"].append(reconstruction_blackbox_consistency(autoencoder, resnet, X_exp_val))
    results["resnet"].append(reconstruction_blackbox_consistency(autoencoder, resnet, X_exp_test))
    results["simplecnn"].append(reconstruction_blackbox_consistency(autoencoder, simplecnn, X_exp_train))
    results["simplecnn"].append(reconstruction_blackbox_consistency(autoencoder, simplecnn, X_exp_val))
    results["simplecnn"].append(reconstruction_blackbox_consistency(autoencoder, simplecnn, X_exp_test))
    results["knn"].append(reconstruction_blackbox_consistency(autoencoder, knn, X_exp_train, keras = False))
    results["knn"].append(reconstruction_blackbox_consistency(autoencoder, knn, X_exp_val, keras = False))
    results["knn"].append(reconstruction_blackbox_consistency(autoencoder, knn, X_exp_test, keras = False))
    
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
    _, _, _, autoencoder = aut.build()
    autoencoder.load_weights("./final_models/cbf/cbf_autoencoder_20191106_150722_best_weights_+1.239848_.hdf5")
    autoencoder_name = "discriminative_autoencoder"
    results[autoencoder_name].append(mean_squared_error(X_exp_train.flatten(), autoencoder.predict(X_exp_train)[0].flatten()))
    results[autoencoder_name].append(np.NaN)
    results[autoencoder_name].append(mean_squared_error(X_exp_val.flatten(), autoencoder.predict(X_exp_val)[0].flatten()))
    results[autoencoder_name].append(np.NaN)
    results[autoencoder_name].append(mean_squared_error(X_exp_test.flatten(), autoencoder.predict(X_exp_test)[0].flatten()))
    results[autoencoder_name].append(np.NaN)
    results[autoencoder_name].append(params["latent_dim"])
    for i in range(12):
        results[autoencoder_name].append(np.NaN)
    results["resnet"].append(reconstruction_blackbox_consistency(autoencoder, resnet, X_exp_train, discriminative = True))
    results["resnet"].append(reconstruction_blackbox_consistency(autoencoder, resnet, X_exp_val, discriminative = True))
    results["resnet"].append(reconstruction_blackbox_consistency(autoencoder, resnet, X_exp_test, discriminative = True))
    results["simplecnn"].append(reconstruction_blackbox_consistency(autoencoder, simplecnn, X_exp_train, discriminative = True))
    results["simplecnn"].append(reconstruction_blackbox_consistency(autoencoder, simplecnn, X_exp_val, discriminative = True))
    results["simplecnn"].append(reconstruction_blackbox_consistency(autoencoder, simplecnn, X_exp_test, discriminative = True))
    results["knn"].append(reconstruction_blackbox_consistency(autoencoder, knn, X_exp_train, keras = False, discriminative = True))
    results["knn"].append(reconstruction_blackbox_consistency(autoencoder, knn, X_exp_val, keras = False, discriminative = True))
    results["knn"].append(reconstruction_blackbox_consistency(autoencoder, knn, X_exp_test, keras = False, discriminative = True))
    
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
    _, _, _, autoencoder = aut.build()
    autoencoder.load_weights("./final_models/cbf/cbf_autoencoder_20191106_153613_best_weights_+1.179660_.hdf5")
    autoencoder_name = "discriminative_vae"
    results[autoencoder_name].append(mean_squared_error(X_exp_train.flatten(), autoencoder.predict(X_exp_train)[0].flatten()))
    results[autoencoder_name].append(np.NaN)
    results[autoencoder_name].append(mean_squared_error(X_exp_val.flatten(), autoencoder.predict(X_exp_val)[0].flatten()))
    results[autoencoder_name].append(np.NaN)
    results[autoencoder_name].append(mean_squared_error(X_exp_test.flatten(), autoencoder.predict(X_exp_test)[0].flatten()))
    results[autoencoder_name].append(np.NaN)
    results[autoencoder_name].append(params["latent_dim"])
    for i in range(12):
        results[autoencoder_name].append(np.NaN)
    results["resnet"].append(reconstruction_blackbox_consistency(autoencoder, resnet, X_exp_train, discriminative = True))
    results["resnet"].append(reconstruction_blackbox_consistency(autoencoder, resnet, X_exp_val, discriminative = True))
    results["resnet"].append(reconstruction_blackbox_consistency(autoencoder, resnet, X_exp_test, discriminative = True))
    results["simplecnn"].append(reconstruction_blackbox_consistency(autoencoder, simplecnn, X_exp_train, discriminative = True))
    results["simplecnn"].append(reconstruction_blackbox_consistency(autoencoder, simplecnn, X_exp_val, discriminative = True))
    results["simplecnn"].append(reconstruction_blackbox_consistency(autoencoder, simplecnn, X_exp_test, discriminative = True))
    results["knn"].append(reconstruction_blackbox_consistency(autoencoder, knn, X_exp_train, keras = False, discriminative = True))
    results["knn"].append(reconstruction_blackbox_consistency(autoencoder, knn, X_exp_val, keras = False, discriminative = True))
    results["knn"].append(reconstruction_blackbox_consistency(autoencoder, knn, X_exp_test, keras = False, discriminative = True))
    
    
    
    
    
    results_df = pd.DataFrame(results, index = results_rows)
    results_df.to_csv("./final_models/" + dataset_name + "_" + time.strftime("%Y%m%d_%H%M%S") + ".csv", sep = ";")
    return results_df



def benchmark_HAR():
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
    
    results = {"resnet":[],
               "simplecnn":[],
               "knn": [],
               "autoencoder": []
              }
    results_rows = ["train_mse", 
                    "train_accuracy",
                    "validation_mse", 
                    "validation_accuracy",
                    "test_mse", 
                    "test_accuracy",
                    "latent_dimension",
                    "reconstruction_blackbox_accuracy_train",
                    "reconstruction_blackbox_accuracy_validation",
                    "reconstruction_blackbox_accuracy_test"
                   ]
    
    resnet = build_resnet(n_timesteps, n_outputs)
    resnet.load_weights("./final_models/HARDataset/HARDataset_blackbox_resnet_20191028_172136_best_weights_+0.99_.hdf5")
    results["resnet"].extend(resnet.evaluate(X_train, y_train))
    results["resnet"].extend(resnet.evaluate(X_val, y_val))
    results["resnet"].extend(resnet.evaluate(X_test, y_test))
    results["resnet"].append(np.NaN)
    
    simplecnn = build_simple_CNN(n_timesteps, n_outputs)
    simplecnn.load_weights("./final_models/HARDataset/HARDataset_blackbox_simpleCNN_20191029_153407_best_weights_+0.94_.hdf5")
    results["simplecnn"].extend(simplecnn.evaluate(X_train, y_train))
    results["simplecnn"].extend(simplecnn.evaluate(X_val, y_val))
    results["simplecnn"].extend(simplecnn.evaluate(X_test, y_test))
    results["simplecnn"].append(np.NaN)
    
    knn = load("./final_models/HARDataset/HARDataset_blackbox_knn_20191031_111540.joblib")
    results["knn"].append(mean_squared_error(y_train, knn.predict(X_train.reshape(X_train.shape[:2]))))
    results["knn"].append(knn.score(X_train.reshape(X_train.shape[:2]), y_train))
    results["knn"].append(mean_squared_error(y_val, knn.predict(X_val.reshape(X_val.shape[:2]))))
    results["knn"].append(knn.score(X_val.reshape(X_val.shape[:2]), y_val))
    results["knn"].append(mean_squared_error(y_test, knn.predict(X_test.reshape(X_test.shape[:2]))))
    results["knn"].append(knn.score(X_test.reshape(X_test.shape[:2]), y_test))
    results["knn"].append(np.NaN)
    
    #params = load("./final_models/HARDataset/HARDataset_autoencoder_20191031_212226_best_weights_.pkl") bugged :(
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
    _, _, autoencoder = aut.build()
    autoencoder.load_weights("./final_models/HARDataset/HARDataset_autoencoder_20191031_212226_best_weights_+0.008519_.hdf5")
    results["autoencoder"].append(autoencoder.evaluate(X_exp_train, X_exp_train))
    results["autoencoder"].append(np.NaN)
    results["autoencoder"].append(autoencoder.evaluate(X_exp_val, X_exp_val))
    results["autoencoder"].append(np.NaN)
    results["autoencoder"].append(autoencoder.evaluate(X_exp_test, X_exp_test))
    results["autoencoder"].append(np.NaN)
    results["autoencoder"].append(params["latent_dim"])
    results["autoencoder"].append(np.NaN)
    results["autoencoder"].append(np.NaN)
    results["autoencoder"].append(np.NaN)
    results["resnet"].append(reconstruction_blackbox_consistency(autoencoder, resnet, X_exp_train))
    results["resnet"].append(reconstruction_blackbox_consistency(autoencoder, resnet, X_exp_val))
    results["resnet"].append(reconstruction_blackbox_consistency(autoencoder, resnet, X_exp_test))
    results["simplecnn"].append(reconstruction_blackbox_consistency(autoencoder, simplecnn, X_exp_train))
    results["simplecnn"].append(reconstruction_blackbox_consistency(autoencoder, simplecnn, X_exp_val))
    results["simplecnn"].append(reconstruction_blackbox_consistency(autoencoder, simplecnn, X_exp_test))
    results["knn"].append(reconstruction_blackbox_consistency(autoencoder, knn, X_exp_train, keras = False))
    results["knn"].append(reconstruction_blackbox_consistency(autoencoder, knn, X_exp_val, keras = False))
    results["knn"].append(reconstruction_blackbox_consistency(autoencoder, knn, X_exp_test, keras = False))
    
    results_df = pd.DataFrame(results, index = results_rows)
    results_df.to_csv("./final_models/" + dataset_name + "_" + time.strftime("%Y%m%d_%H%M%S") + ".csv", sep = ";")
    return results_df

def benchmark_phalanges():
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
    
    results = {"resnet":[],
               "simplecnn":[],
               "knn": [],
               "autoencoder": []
              }
    results_rows = ["train_mse", 
                    "train_accuracy",
                    "validation_mse", 
                    "validation_accuracy",
                    "test_mse", 
                    "test_accuracy",
                    "latent_dimension",
                    "reconstruction_blackbox_accuracy_train",
                    "reconstruction_blackbox_accuracy_validation",
                    "reconstruction_blackbox_accuracy_test"
                   ]
    
    resnet = build_resnet(n_timesteps, n_outputs)
    resnet.load_weights("./final_models/phalanges/phalanges_blackbox_resnet_20191101_164247_best_weights_+0.86_.hdf5")
    results["resnet"].extend(resnet.evaluate(X_train, y_train))
    results["resnet"].extend(resnet.evaluate(X_val, y_val))
    results["resnet"].extend(resnet.evaluate(X_test, y_test))
    results["resnet"].append(np.NaN)
    
    simplecnn = build_simple_CNN(n_timesteps, n_outputs)
    simplecnn.load_weights("./final_models/phalanges/phalanges_blackbox_simpleCNN_20191101_170209_best_weights_+0.83_.hdf5")
    results["simplecnn"].extend(simplecnn.evaluate(X_train, y_train))
    results["simplecnn"].extend(simplecnn.evaluate(X_val, y_val))
    results["simplecnn"].extend(simplecnn.evaluate(X_test, y_test))
    results["simplecnn"].append(np.NaN)
    
    knn = load("./final_models/phalanges/phalanges_blackbox_knn_20191101_171534.joblib")
    results["knn"].append(mean_squared_error(y_train, knn.predict(X_train.reshape(X_train.shape[:2]))))
    results["knn"].append(knn.score(X_train.reshape(X_train.shape[:2]), y_train))
    results["knn"].append(mean_squared_error(y_val, knn.predict(X_val.reshape(X_val.shape[:2]))))
    results["knn"].append(knn.score(X_val.reshape(X_val.shape[:2]), y_val))
    results["knn"].append(mean_squared_error(y_test, knn.predict(X_test.reshape(X_test.shape[:2]))))
    results["knn"].append(knn.score(X_test.reshape(X_test.shape[:2]), y_test))
    results["knn"].append(np.NaN)
    
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
    _, _, autoencoder = aut.build()
    autoencoder.load_weights("./final_models/phalanges/phalanges_autoencoder_20191103_211535_best_weights_+0.0010_.hdf5")
    results["autoencoder"].append(autoencoder.evaluate(X_exp_train, X_exp_train))
    results["autoencoder"].append(np.NaN)
    results["autoencoder"].append(autoencoder.evaluate(X_exp_val, X_exp_val))
    results["autoencoder"].append(np.NaN)
    results["autoencoder"].append(autoencoder.evaluate(X_exp_test, X_exp_test))
    results["autoencoder"].append(np.NaN)
    results["autoencoder"].append(params["latent_dim"])
    results["autoencoder"].append(np.NaN)
    results["autoencoder"].append(np.NaN)
    results["autoencoder"].append(np.NaN)
    results["resnet"].append(reconstruction_blackbox_consistency(autoencoder, resnet, X_exp_train))
    results["resnet"].append(reconstruction_blackbox_consistency(autoencoder, resnet, X_exp_val))
    results["resnet"].append(reconstruction_blackbox_consistency(autoencoder, resnet, X_exp_test))
    results["simplecnn"].append(reconstruction_blackbox_consistency(autoencoder, simplecnn, X_exp_train))
    results["simplecnn"].append(reconstruction_blackbox_consistency(autoencoder, simplecnn, X_exp_val))
    results["simplecnn"].append(reconstruction_blackbox_consistency(autoencoder, simplecnn, X_exp_test))
    results["knn"].append(reconstruction_blackbox_consistency(autoencoder, knn, X_exp_train, keras = False))
    results["knn"].append(reconstruction_blackbox_consistency(autoencoder, knn, X_exp_val, keras = False))
    results["knn"].append(reconstruction_blackbox_consistency(autoencoder, knn, X_exp_test, keras = False))
    results_df = pd.DataFrame(results, index = results_rows)
    results_df.to_csv("./final_models/" + dataset_name + "_" + time.strftime("%Y%m%d_%H%M%S") + ".csv", sep = ";")
    return results_df