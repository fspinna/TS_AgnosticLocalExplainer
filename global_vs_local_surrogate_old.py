#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 11:16:22 2019

@author: francesco
"""

from agnosticlocalexplainer import AgnosticLocalExplainer
import numpy as np

def shapelet_local_explain(blackbox, 
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
                           l=0.1, 
                           r=2, 
                           weight_regularizer=.01, 
                           optimizer="sgd", 
                           max_iter=100
                           ):
    Y_blackbox_original = []    # blackbox prediction of original ts
    Y_blackbox_original_proba = []  # blackbox proba prediction
    Y_blackbox_reconstructed = []   # blackbox prediction of autoencoder reconstructed ts
    Y_blackbox_reconstructed_proba = [] # blackbox proba prediction of autoencoder reconstructed ts
    Y_surrogate_original = []   # surrogate prediction of original ts
    Y_surrogate_original_proba = [] # surrogate proba prediction of original ts
    Y_surrogate_reconstructed = []  # surrogate prediction of autoencoder reconstructed ts
    Y_surrogate_reconstructed_proba = []    # surrogate proba prediction of autoencoder reconstructed ts
    shapelet_explainers = []
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
        shapelet_explainer = agnostic.build_shapelet_explainer(l=l, r=r, weight_regularizer=weight_regularizer, optimizer=optimizer, max_iter=max_iter)
        
        y_blackbox_original = agnostic.blackbox_predict(agnostic.instance_to_explain.reshape(1,-1,1))[0]
        y_blackbox_original_proba = agnostic.blackbox_predict_proba(agnostic.instance_to_explain.reshape(1,-1,1))[0]
        
        y_blackbox_reconstructed = agnostic.blackbox_decode_and_predict(agnostic.instance_to_explain_latent.reshape(1,-1))[0]
        y_blackbox_reconstructed_proba = agnostic.blackbox_decode_and_predict_proba(agnostic.instance_to_explain_latent.reshape(1,-1))[0]
        
        y_surrogate_original = shapelet_explainer.predict(agnostic.instance_to_explain.reshape(1,-1))[0]
        y_surrogate_original_proba = shapelet_explainer.predict_proba(agnostic.instance_to_explain.reshape(1,-1))[0]
        
        y_surrogate_reconstructed = shapelet_explainer.predict(decoder.predict(agnostic.instance_to_explain_latent.reshape(1,-1)))[0]
        y_surrogate_reconstructed_proba = shapelet_explainer.predict_proba(decoder.predict(agnostic.instance_to_explain_latent.reshape(1,-1)))[0]
        
        Y_blackbox_original.append(y_blackbox_original)
        Y_blackbox_reconstructed.append(y_blackbox_reconstructed)
        Y_surrogate_original.append(y_surrogate_original)
        Y_surrogate_reconstructed.append(y_surrogate_reconstructed)
        
        Y_blackbox_original_proba.append(y_blackbox_original_proba)
        Y_blackbox_reconstructed_proba.append(y_blackbox_reconstructed_proba)
        Y_surrogate_original_proba.append(y_surrogate_original_proba)
        Y_surrogate_reconstructed_proba.append(y_surrogate_reconstructed_proba)
        
        shapelet_explainers.append(shapelet_explainer)
        agnostic_explainers.append(agnostic)
        
        print(index_to_explain + 1, "/", len(X_explanation))
        
    
    return {"y_blackbox_original": np.array(Y_blackbox_original), 
            "y_blackbox_reconstructed": np.array(Y_blackbox_reconstructed), 
            "y_surrogate_original": np.array(Y_surrogate_original), 
            "y_surrogate_reconstructed": np.array(Y_surrogate_reconstructed), 
            "y_blackbox_original_proba": np.array(Y_blackbox_original_proba), 
            "y_blackbox_reconstructed_proba": np.array(Y_blackbox_reconstructed_proba), 
            "y_surrogate_original_proba": np.array(Y_surrogate_original_proba), 
            "y_surrogate_reconstructed_proba": np.array(Y_surrogate_reconstructed_proba), 
            "shapelet_explainers": shapelet_explainers,
            "agnostic_explainers": agnostic_explainers
            }
    
    
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
    
    
    knn = load("./blackbox_checkpoints/cbf_blackbox_knn_20191106_145654.joblib")
    
    blackbox = build_resnet(n_timesteps, n_outputs)
    blackbox.load_weights("./blackbox_checkpoints/cbf_blackbox_resnet_20191106_145242_best_weights_+1.00_.hdf5")
    resnet = blackbox
    
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
    
    
    blackbox = resnet
    encoder = autoencoder.layers[1]
    decoder = autoencoder.layers[2]
    blackbox_input_dimensions = 3
    labels = ["cylinder", "bell", "funnel"]
    
    
    blackbox_predict = BlackboxPredictWrapper(blackbox, 3)
    args = {"max_iter": 100}
    global_surrogate = AgnosticGlobalExplainer(**args)
    global_surrogate.fit(X_exp_train[:,:,0], blackbox_predict.predict(X_exp_train))
    y_blackbox = blackbox_predict.predict(X_exp_test)
    y_blackbox_proba = blackbox_predict.predict_proba(X_exp_test)
    y_surrogate = global_surrogate.predict(X_exp_test[:,:,0])
    y_surrogate_proba = global_surrogate.predict_proba(X_exp_test[:,:,0])
    
    
    local_results = shapelet_local_explain(blackbox, 
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
    
    
    results = {"y_blackbox_original_LOCAL": local_results["y_blackbox_original"], 
               "y_blackbox_reconstructed_LOCAL": local_results["y_blackbox_reconstructed"], 
               "y_surrogate_original_LOCAL": local_results["y_surrogate_original"], 
               "y_surrogate_reconstructed_LOCAL": local_results["y_surrogate_reconstructed"],
               "y_blackbox_GLOBAL": y_blackbox,
               "y_surrogate_GLOBAL": y_surrogate,
               "y_blackbox_original_LOCAL_proba": local_results["y_blackbox_original_proba"], 
               "y_blackbox_reconstructed_LOCAL_proba": local_results["y_blackbox_reconstructed_proba"], 
               "y_surrogate_original_LOCAL_proba": local_results["y_surrogate_original_proba"], 
               "y_surrogate_reconstructed_LOCAL_proba": local_results["y_surrogate_reconstructed_proba"],
               "y_blackbox_GLOBAL_proba": y_blackbox_proba,
               "y_surrogate_GLOBAL_proba": y_surrogate_proba
               }
    results_df = results#pd.DataFrame(results)
    
    local_classification_report = classification_report(results_df["y_blackbox_original_LOCAL"], results_df["y_surrogate_reconstructed_LOCAL"])
    global_classification_report = classification_report(results_df["y_blackbox_GLOBAL"], results_df["y_surrogate_GLOBAL"])
    reconstrution_classification_report = classification_report(results_df["y_blackbox_original_LOCAL"], results_df["y_blackbox_reconstructed_LOCAL"])
    
    local_fidelity = accuracy_score(results_df["y_blackbox_original_LOCAL"], results_df["y_surrogate_reconstructed_LOCAL"])
    global_fidelity = accuracy_score(results_df["y_blackbox_GLOBAL"], results_df["y_surrogate_GLOBAL"])
    reconstruction_fidelity = accuracy_score(results_df["y_blackbox_original_LOCAL"], results_df["y_blackbox_reconstructed_LOCAL"])
    
    local_coverage = coverage_error(results_df["y_blackbox_reconstructed_LOCAL_proba"].round(), results_df["y_surrogate_reconstructed_LOCAL_proba"])
    global_coverage = coverage_error(results_df["y_blackbox_GLOBAL_proba"].round(), results_df["y_surrogate_GLOBAL_proba"])
    
    print("local fidelity: ", local_fidelity)
    print("global fidelity: ", global_fidelity)
    print("reconstruction fidelity: ", reconstruction_fidelity)
    
    print("local coverage: ", local_coverage)
    print("global coverage: ", global_coverage)
    
    file_path = "./agnostic_explainers/" + dataset_name + "_agnosticvsglobal_" + time.strftime("%Y%m%d_%H%M%S") + ".joblib"
    dump(results, file_path)
    
    
    
