#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 11:16:22 2019

@author: francesco
"""

from agnosticlocalexplainer import AgnosticLocalExplainer
import numpy as np

def build_neighborhoods(blackbox, 
                           encoder, 
                           decoder, 
                           autoencoder, 
                           X_explanation, 
                           y_explanation,
                           blackbox_input_dimensions,
                           labels = None,
                           size = 100,
                           neigh_type = "rndgen",
                           ngen = 10,
                           ):
    
    agnostic_explainers = []
    for index_to_explain in range(len(X_explanation)):
        agnostic = AgnosticLocalExplainer(blackbox, 
                                      encoder, 
                                      decoder, 
                                      autoencoder,  
                                      X_explanation = X_explanation, 
                                      y_explanation = y_explanation, 
                                      index_to_explain = index_to_explain,
                                      blackbox_input_dimensions = blackbox_input_dimensions,
                                      labels = labels
                                     )
        #agnostic.check_autoencoder_blackbox_consistency()
        print("\nNeighborhood Generation")
        agnostic.LOREM_neighborhood_generation(
                              neigh_type = neigh_type, 
                              categorical_use_prob = True,
                              continuous_fun_estimation = False, 
                              size = size,
                              ocr = 0.1, 
                              multi_label=False,
                              one_vs_rest=False,
                              verbose = True,
                              ngen = ngen)
        agnostic.LOREM_tree_rules_extraction()
        agnostic_explainers.append(agnostic)
    return agnostic_explainers

def build_local_shapelet_trees(agnostic_explainers,
                                      l=0.1, 
                                      r=2, 
                                      weight_regularizer=.01, 
                                      optimizer="sgd", 
                                      max_iter=100,
                                      random_state = None
                                      ):
        
    Y_blackbox_original = []    # blackbox prediction of original ts
    Y_blackbox_reconstructed = []   # blackbox prediction of autoencoder reconstructed ts
    Y_surrogate_original = []   # surrogate prediction of original ts
    Y_surrogate_reconstructed = []  # surrogate prediction of autoencoder reconstructed ts
    Y_LORE = [] # LORE tree prediction
    fidelity_LORE_LOCAL = [] # LORE tree fidelity
    fidelity_neighborhood_shapelet_LOCAL = [] # internal shapelet tree fidelity (inside the agnostic explainer)
    shapelet_explainers = []
    counter = 0
    for agnostic in agnostic_explainers:
        shapelet_explainer = agnostic.build_shapelet_explainer(l=l, 
                                                               r=r, 
                                                               weight_regularizer=weight_regularizer, 
                                                               optimizer=optimizer, 
                                                               max_iter=max_iter,
                                                               random_state = random_state
                                                               )
        
        y_blackbox_original = agnostic.blackbox_predict(agnostic.instance_to_explain.reshape(1,-1,1))[0]
        y_blackbox_reconstructed = agnostic.blackbox_decode_and_predict(agnostic.instance_to_explain_latent.reshape(1,-1))[0]
        y_surrogate_original = shapelet_explainer.predict(agnostic.instance_to_explain.reshape(1,-1))[0]
        y_surrogate_reconstructed = shapelet_explainer.predict(decoder.predict(agnostic.instance_to_explain_latent.reshape(1,-1)))[0]
        y_LORE = agnostic.LOREM_Explanation.dt_pred
        fidelity_LORE = agnostic.LOREM_Explanation.fidelity
        fidelity_neighborhood_shapelet = agnostic.shapelet_explainer.fidelity
        
        Y_blackbox_original.append(y_blackbox_original)
        Y_blackbox_reconstructed.append(y_blackbox_reconstructed)
        Y_surrogate_original.append(y_surrogate_original)
        Y_surrogate_reconstructed.append(y_surrogate_reconstructed)
        Y_LORE.append(y_LORE)
        fidelity_LORE_LOCAL.append(fidelity_LORE)
        fidelity_neighborhood_shapelet_LOCAL.append(fidelity_neighborhood_shapelet)
        
        shapelet_explainers.append(shapelet_explainer)
        
        counter += 1 
        print(counter, "/", len(agnostic_explainers))
        
    return {"y_blackbox_original_LOCAL": np.array(Y_blackbox_original), 
            "y_blackbox_reconstructed_LOCAL": np.array(Y_blackbox_reconstructed), 
            "y_surrogate_original_LOCAL": np.array(Y_surrogate_original), 
            "y_surrogate_reconstructed_LOCAL": np.array(Y_surrogate_reconstructed),
            "y_LORE_LOCAL": np.array(Y_LORE),
            "fidelity_LORE_LOCAL": np.array(fidelity_LORE_LOCAL),
            "fidelity_neighborhood_shapelet_LOCAL": np.array(fidelity_neighborhood_shapelet_LOCAL)
            }, shapelet_explainers
    
    
if __name__ == "__main__":
    
    from agnosticglobalexplainer import AgnosticGlobalExplainer
    from myutils import BlackboxPredictWrapper
    import keras
    from pyts.datasets import make_cylinder_bell_funnel
    from sklearn.model_selection import train_test_split
    from autoencoders import Autoencoder
    from joblib import load, dump
    from blackboxes import build_resnet
    from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, coverage_error
    import pandas as pd
    import time
    
    random_state = 0
    dataset_name = "cbf"
    
    
    X_all, y_all = make_cylinder_bell_funnel(n_samples = 600, random_state = random_state)
    X_all = X_all.reshape((X_all.shape[0], X_all.shape[1], 1))
    
    print("DATASET INFO:")
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
    
    print("\nSHAPES:")
    print("BLACKBOX TRAINING SET: ", X_train.shape)
    print("BLACKBOX VALIDATION SET: ", X_val.shape)
    print("BLACKBOX TEST SET: ", X_test.shape)
    print("EXPLANATION TRAINING SET: ", X_exp_train.shape)
    print("EXPLANATION VALIDATION SET: ", X_exp_val.shape)
    print("EXPLANATION TEST SET: ", X_exp_test.shape)
    
    n_timesteps, n_outputs, n_features = X_train.shape[1], len(np.unique(y_all)), 1 
    print("\nTIMESTEPS: ", n_timesteps)
    print("N. LABELS: ", n_outputs)
    
    
    blackbox = build_resnet(n_timesteps, n_outputs)
    blackbox.load_weights("./blackbox_checkpoints/cbf_blackbox_resnet_20191106_145242_best_weights_+1.00_.hdf5")
    resnet = blackbox
    """
    params = {"input_shape": (n_timesteps,1),
          "n_blocks": 8, 
          "latent_dim": 2,
          "encoder_latent_layer_type": "variational",
          "encoder_args": {"filters":[2,4,8,16,32,64,128,256], 
                            "kernel_size":[21,18,15,13,11,8,5,3], 
                            "padding":"same", 
                            "activation":"elu", 
                            "pooling":[1,1,1,1,1,1,1,1]}
         }

    aut = Autoencoder(verbose = False, **params)
    encoder, decoder, autoencoder = aut.build()
    autoencoder.load_weights("./autoencoder_checkpoints/cbf_autoencoder_20191106_144909_best_weights_+136.8745_.hdf5")
    """
    params = {"input_shape": (n_timesteps,1),
          "n_blocks": 8, 
          "latent_dim": 2,
          "encoder_latent_layer_type": "dense",
          "encoder_args": {"filters":[2,4,8,16,32,64,128,256], 
                            "kernel_size":[21,18,15,13,11,8,5,3], 
                            "padding":"same", 
                            "activation":"elu", 
                            "pooling":[1,1,1,1,1,1,1,1]}
         }

    aut = Autoencoder(verbose = False, **params)
    encoder, decoder, autoencoder = aut.build()
    autoencoder.load_weights("./autoencoder_checkpoints/cbf_autoencoder_20191106_144056_best_weights_+1.0504_.hdf5")
    
    blackbox = resnet
    encoder = autoencoder.layers[1]
    decoder = autoencoder.layers[2]
    blackbox_input_dimensions = 3
    labels = ["cylinder", "bell", "funnel"]
    
    
    agnostic_explainers = build_neighborhoods(blackbox, 
                           encoder, 
                           decoder, 
                           autoencoder, 
                           X_exp_test,
                           y_exp_test, 
                           blackbox_input_dimensions, 
                           ngen = 10, 
                           size = 1000, 
                           neigh_type = "geneticp",
                           labels = ["cylinder", "bell", "funnel"])
    
    file_path = "./agnostic_explainers/" + dataset_name + "_" + time.strftime("%Y%m%d_%H%M%S")
    dump(agnostic_explainers, file_path + "_local.pkl")
    
    blackbox_predict = BlackboxPredictWrapper(blackbox, 3)
    global_surrogate = AgnosticGlobalExplainer(random_state = random_state, max_iter = 50)
    global_surrogate.fit(X_exp_train[:,:,0], blackbox_predict.predict(X_exp_train))
    y_blackbox_original_GLOBAL = blackbox_predict.predict(X_exp_test)
    y_blackbox_reconstructed_GLOBAL = blackbox_predict.predict(decoder.predict(encoder.predict(X_exp_test)))
    y_surrogate_original_GLOBAL = global_surrogate.predict(X_exp_test[:,:,0])
    y_surrogate_reconstructed_GLOBAL = global_surrogate.predict(decoder.predict(encoder.predict(X_exp_test))[:,:,0])
    global_results = {"y_blackbox_original_GLOBAL":y_blackbox_original_GLOBAL,
                      "y_blackbox_reconstructed_GLOBAL":y_blackbox_reconstructed_GLOBAL,
                      "y_surrogate_original_GLOBAL":y_surrogate_original_GLOBAL,
                      "y_surrogate_reconstructed_GLOBAL":y_surrogate_reconstructed_GLOBAL
                      }
    local_results, shapelet_explainers = build_local_shapelet_trees(agnostic_explainers, 
                                                                    random_state = random_state, 
                                                                    max_iter = 50)
    
    results = {**local_results, **global_results}
    results_df = pd.DataFrame(results)
    results_df.to_csv(file_path + ".csv", sep = ";", index = False)
    
    local_classification_report = classification_report(results_df["y_blackbox_original_LOCAL"], results_df["y_surrogate_reconstructed_LOCAL"])
    global_classification_report = classification_report(results_df["y_blackbox_original_GLOBAL"], results_df["y_surrogate_original_GLOBAL"])
    reconstrution_classification_report = classification_report(results_df["y_blackbox_original_LOCAL"], results_df["y_blackbox_reconstructed_LOCAL"])
    
    local_fidelity = accuracy_score(results_df["y_blackbox_original_LOCAL"], results_df["y_surrogate_reconstructed_LOCAL"])
    global_fidelity = accuracy_score(results_df["y_blackbox_original_GLOBAL"], results_df["y_surrogate_original_GLOBAL"])
    reconstruction_fidelity = accuracy_score(results_df["y_blackbox_original_LOCAL"], results_df["y_blackbox_reconstructed_LOCAL"])

    
    print("local fidelity: ", local_fidelity)
    print("global fidelity: ", global_fidelity)
    print("reconstruction fidelity: ", reconstruction_fidelity)
    
    
    
    
