#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 11:16:22 2019

@author: francesco
"""

from agnosticlocalexplainer import AgnosticLocalExplainer, save_agnostic_local_explainer, load_agnostic_local_explainer
from agnosticglobalexplainer import AgnosticGlobalExplainer, save_shapelet_model, load_shapelet_model, extract_used_shapelets, shapelet_stability_measure
import numpy as np
from joblib import load, dump
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, coverage_error
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist
#from shap_utils import shap_ts, shap_output_to_point_by_point, shap_f
    
def build_agnostic_local_explainers(blackbox, 
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
                                  max_iter=100,
                                  random_state = None,
                                  distance_quantile_threshold=np.array(list(range(1,10)))/10
                                   ):
    agnostic_explainers = []
    counter = 0
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
                              filter_crules = False,
                              ngen = ngen)
        agnostic.LOREM_tree_rules_extraction()
        agnostic.build_rules_dataframes()
        if len(agnostic.rules_dataframes.keys()) == 1: 
            print()
            print("NO CRULES!!!")
            print()
        agnostic.rules_check_by_augmentation(num_samples = 1000, remove_bad = True, keep_one_crule = True)
        agnostic.build_shapelet_explainer(l=l, 
                                           r=r, 
                                           weight_regularizer=weight_regularizer, 
                                           optimizer=optimizer, 
                                           max_iter=max_iter,
                                           random_state = random_state,
                                           distance_quantile_threshold = distance_quantile_threshold
                                           )
        agnostic_explainers.append(agnostic)
        counter += 1 
        print(counter, "/", len(X_explanation))
    return agnostic_explainers

def save_agnostic_local_explainers(agnostic_explainers, file_path, verbose = False):
    folder = file_path + "/"
    for i, agnostic in enumerate(agnostic_explainers):
        save_agnostic_local_explainer(agnostic, folder + "_" + str(i) + "_")
        if verbose:
            print(i+1, "/", len(agnostic_explainers))
        
def load_agnostic_local_explainers(file_path, n_explainers, verbose = False):
    folder = file_path + "/"
    agnostic_explainers = []
    for i in range(n_explainers):
        agnostic = load_agnostic_local_explainer(folder + "_" + str(i) + "_")
        agnostic_explainers.append(agnostic)
        if verbose:
            print(i+1, "/", n_explainers)
    return agnostic_explainers
        
def massive_save_agnostic_local_explainers(agnostic_explainers, file_path, verbose = False):
    folder = file_path + "/"
    for i, agnostic in enumerate(agnostic_explainers):
        save_shapelet_model(agnostic.shapelet_explainer, folder + "_" + str(i) + "_")
        agnostic.shapelet_explainer = None
        if verbose:
            print(i+1, "/", len(agnostic_explainers))
    dump(agnostic_explainers, file_path + "/" + "agnostic_explainers.pkl")
    
def massive_load_agnostic_local_explainers(file_path, verbose = False, load_shapelet_explainers = True):
    folder = file_path + "/"
    agnostic_explainers = load(file_path + "/" + "agnostic_explainers.pkl")
    if load_shapelet_explainers:
        for i, agnostic in enumerate(agnostic_explainers):
            agnostic.shapelet_explainer = load_shapelet_model(folder + "_" + str(i) + "_")
            if verbose:
                print(i+1, "/", len(agnostic_explainers))
    return agnostic_explainers
    

def get_local_predictions(agnostic_explainers):
    Y_blackbox_original = []    # blackbox prediction of original ts
    Y_blackbox_reconstructed = []   # blackbox prediction of autoencoder reconstructed ts
    Y_surrogate_original = []   # surrogate prediction of original ts
    Y_surrogate_reconstructed = []  # surrogate prediction of autoencoder reconstructed ts
    Y_LORE = [] # LORE tree prediction
    fidelity_LORE_LOCAL = [] # LORE tree fidelity
    coverage_LORE_LOCAL = [] # LORE rule coverage
    precision_LORE_LOCAL = [] # LORE rule precision
    fidelity_neighborhood_shapelet_LOCAL = [] # internal shapelet tree fidelity (inside the agnostic explainer)
    coverage_shapelet_LOCAL = [] # internal shapelet rule coverage
    precision_shapelet_LOCAL = [] # internal shapelet rule precision
    for agnostic in agnostic_explainers:
        y_blackbox_original = agnostic.blackbox_predict(agnostic.instance_to_explain.reshape(1,-1,1))[0]
        y_blackbox_reconstructed = agnostic.blackbox_decode_and_predict(agnostic.instance_to_explain_latent.reshape(1,-1))[0]
        y_surrogate_original = agnostic.shapelet_explainer.predict(agnostic.instance_to_explain.reshape(1,-1))[0]
        y_surrogate_reconstructed = agnostic.shapelet_explainer.predict(agnostic.decoder.predict(agnostic.instance_to_explain_latent.reshape(1,-1)))[0]
        y_LORE = agnostic.LOREM_Explanation.dt_pred
        fidelity_LORE = agnostic.LOREM_Explanation.fidelity
        coverage_LORE = agnostic.LOREM_coverage
        precision_LORE = agnostic.LOREM_precision
        fidelity_neighborhood_shapelet = agnostic.shapelet_explainer.fidelity
        coverage_shapelet = agnostic.shapelet_explainer.coverage_score(agnostic.decoder.predict(agnostic.instance_to_explain_latent.reshape(1,-1)).ravel())
        precision_shapelet = agnostic.shapelet_explainer.precision_score(agnostic.decoder.predict(agnostic.instance_to_explain_latent.reshape(1,-1)).ravel(),
                                                                         agnostic.Zy_latent_instance_neighborhood_labels)
        
        Y_blackbox_original.append(y_blackbox_original)
        Y_blackbox_reconstructed.append(y_blackbox_reconstructed)
        Y_surrogate_original.append(y_surrogate_original)
        Y_surrogate_reconstructed.append(y_surrogate_reconstructed)
        Y_LORE.append(y_LORE)
        fidelity_LORE_LOCAL.append(fidelity_LORE)
        coverage_LORE_LOCAL.append(coverage_LORE)
        precision_LORE_LOCAL.append(precision_LORE)
        fidelity_neighborhood_shapelet_LOCAL.append(fidelity_neighborhood_shapelet)
        coverage_shapelet_LOCAL.append(coverage_shapelet)
        precision_shapelet_LOCAL.append(precision_shapelet)
    return {"y_blackbox_original_LOCAL": np.array(Y_blackbox_original), 
            "y_blackbox_reconstructed_LOCAL": np.array(Y_blackbox_reconstructed), 
            "y_surrogate_original_LOCAL": np.array(Y_surrogate_original), 
            "y_surrogate_reconstructed_LOCAL": np.array(Y_surrogate_reconstructed),
            "y_LORE_LOCAL": np.array(Y_LORE),
            "coverage_LORE_LOCAL": np.array(coverage_LORE_LOCAL),
            "precision_LORE_LOCAL": np.array(precision_LORE_LOCAL),
            "fidelity_LORE_LOCAL": np.array(fidelity_LORE_LOCAL),
            "fidelity_neighborhood_shapelet_LOCAL": np.array(fidelity_neighborhood_shapelet_LOCAL),
            "coverage_shapelet_LOCAL": np.array(coverage_shapelet_LOCAL),
            "precision_shapelet_LOCAL": np.array(precision_shapelet_LOCAL)
            }

def get_global_predictions(global_surrogate, blackbox_predict, dataset, y_blackbox_train, encoder, decoder):
    y_blackbox_original_GLOBAL = blackbox_predict.predict(dataset)
    y_blackbox_reconstructed_GLOBAL = blackbox_predict.predict(decoder.predict(encoder.predict(dataset)))
    y_surrogate_original_GLOBAL = global_surrogate.predict(dataset[:,:,0])
    y_surrogate_reconstructed_GLOBAL = global_surrogate.predict(decoder.predict(encoder.predict(dataset))[:,:,0])
    coverage_shapelet_GLOBAL = []
    precision_shapelet_GLOBAL = []
    for ts in dataset:
        coverage_shapelet_GLOBAL.append(global_surrogate.coverage_score(ts))
        precision_shapelet_GLOBAL.append(global_surrogate.precision_score(ts, y_blackbox_train))
    global_results = {"y_blackbox_original_GLOBAL":y_blackbox_original_GLOBAL,
                      "y_blackbox_reconstructed_GLOBAL":y_blackbox_reconstructed_GLOBAL,
                      "y_surrogate_original_GLOBAL":y_surrogate_original_GLOBAL,
                      "y_surrogate_reconstructed_GLOBAL":y_surrogate_reconstructed_GLOBAL,
                      "coverage_shapelet_GLOBAL": np.array(coverage_shapelet_GLOBAL),
                      "precision_shapelet_GLOBAL": np.array(precision_shapelet_GLOBAL)
                      }
    return global_results

def get_all_predictions(agnostic_explainers, global_surrogate, blackbox_predict, dataset, y_blackbox_train, encoder, decoder):
    local_results = get_local_predictions(agnostic_explainers)
    global_results = get_global_predictions(global_surrogate, blackbox_predict, dataset, y_blackbox_train, encoder, decoder)
    #stability = {"stability": calculate_stability(agnostic_explainers, dataset, encoder)}
    results = {**local_results, **global_results}#, **stability}
    results_df = pd.DataFrame(results)
    return results_df

"""
def calculate_stability(agnostic_explainers, X_exp_test, encoder, n_neighbors = 4):
    X_exp_test_latent = encoder.predict(X_exp_test)
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(X_exp_test_latent)
    distances, indices = nbrs.kneighbors(X_exp_test_latent)
    stabilities = []
    for i, agnostic in enumerate(agnostic_explainers):
        neighborhood_distances = np.array(distances[i][1:]) # the first value is the distance from the point itself (0)
        neighborhood_indices = indices[i][1:]
        instance_exemplars = agnostic.rules_dataframes["rule"]["df"]
        neighborhood_average_exemplar_distances = []
        for index in neighborhood_indices:
            neighbor_exemplars = agnostic_explainers[index].rules_dataframes["rule"]["df"]
            neighbor_average_exemplar_distance = cdist(instance_exemplars, neighbor_exemplars).mean()
            neighborhood_average_exemplar_distances.append(neighbor_average_exemplar_distance)
        neighborhood_average_exemplar_distances = np.array(neighborhood_average_exemplar_distances)
        stability = (neighborhood_average_exemplar_distances/neighborhood_distances).max()
        stabilities.append(stability)
    return np.array(stabilities)
"""

"""
def calculate_stability(agnostic_explainers, X_exp_test, encoder, n_neighbors = 4):
    #X_exp_test_latent = encoder.predict(X_exp_test)
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(X_exp_test)
    distances, indices = nbrs.kneighbors(X_exp_test)
    stabilities = []
    for i, agnostic in enumerate(agnostic_explainers):
        neighborhood_distances = np.array(distances[i][1:]) # the first value is the distance from the point itself (0)
        neighborhood_indices = indices[i][1:]
        instance_exemplars = agnostic.rules_dataframes["rule"]["df"]
        neighborhood_average_exemplar_distances = []
        for index in neighborhood_indices:
            neighbor_exemplars = agnostic_explainers[index].rules_dataframes["rule"]["df"]
            neighbor_average_exemplar_distance = cdist(instance_exemplars, neighbor_exemplars).mean()
            neighborhood_average_exemplar_distances.append(neighbor_average_exemplar_distance)
        neighborhood_average_exemplar_distances = np.array(neighborhood_average_exemplar_distances)
        stability = (neighborhood_average_exemplar_distances/neighborhood_distances).max()
        stabilities.append(stability)
    return np.array(stabilities)
"""
"""           
def calculate_shapelet_stability(agnostic_explainers, X_exp_test, n_neighbors = 4, quantile = 0.9):
    X_exp_test = X_exp_test[:,:,0]
    encoder = agnostic_explainers[0].encoder
    decoder = agnostic_explainers[0].decoder
    nbrs = NearestNeighbors(n_neighbors=len(X_exp_test), algorithm='ball_tree').fit(X_exp_test)
    distances, indices = nbrs.kneighbors(X_exp_test)
    stabilities = []
    for i, agnostic in enumerate(agnostic_explainers):
        ts = decoder.predict(agnostic.instance_to_explain_latent.reshape(1,-1)).ravel()
        all_neighborhood_indices = indices[i][1:]
        nearest_neighbors_indices = indices[i][1:n_neighbors]
        
        far_neighbor_index = all_neighborhood_indices[np.quantile(range(len(X_exp_test)-1), quantile, interpolation = "lower")]
        far_neighbor = decoder.predict(encoder.predict(X_exp_test[far_neighbor_index].reshape(1,-1,1))).ravel()
        
        # SHAPELETS FOR THE INSTANCE TO EXPLAIN
        shapelets = extract_used_shapelets(agnostic.shapelet_explainer, ts)
        
        
        # SHAPELETS FOR THE FAR NEIGHBOR
        far_shapelets = extract_used_shapelets(agnostic_explainers[far_neighbor_index].shapelet_explainer, far_neighbor)
        
        
        far_sim = shapelet_stability_measure(shapelets, far_shapelets)
        print()
        print("far:",far_sim)
        
        stability = []
        
        # SHAPELETS FOR THE K-NEAREST NEIGHBORS
        for index in nearest_neighbors_indices:
            neighbor = decoder.predict(encoder.predict(X_exp_test[index].reshape(1,-1,1))).ravel()
            near_shapelets = extract_used_shapelets(agnostic_explainers[index].shapelet_explainer, neighbor)
            sim = shapelet_stability_measure(shapelets, near_shapelets)
            print("near:",sim)
            stability.append(sim)
        stability = np.array(stability)/far_sim
        stabilities.append(np.array([stability.max(), stability.min(), stability.mean()]))
    return np.array(stabilities)
"""
"""
def calculate_shapelet_stability(agnostic_explainers, X_exp_test, n_neighbors = 6):
    X_exp_test = X_exp_test[:,:,0]
    encoder = agnostic_explainers[0].encoder
    decoder = agnostic_explainers[0].decoder
    nbrs = NearestNeighbors(n_neighbors=len(X_exp_test), algorithm='ball_tree').fit(X_exp_test)
    distances, indices = nbrs.kneighbors(X_exp_test)
    stabilities = []
    for i, agnostic in enumerate(agnostic_explainers):
        ts = decoder.predict(agnostic.instance_to_explain_latent.reshape(1,-1)).ravel()
        nearest_neighbors_indices = indices[i][1:n_neighbors]
        
        far_neighbor_index = nearest_neighbors_indices[-1]
        far_neighbor = decoder.predict(encoder.predict(X_exp_test[far_neighbor_index].reshape(1,-1,1))).ravel()
        
        nearest_neighbor_index = nearest_neighbors_indices[0]
        nearest_neighbor = decoder.predict(encoder.predict(X_exp_test[nearest_neighbor_index].reshape(1,-1,1))).ravel()
        
        # SHAPELETS FOR THE INSTANCE TO EXPLAIN
        shapelets = extract_used_shapelets(agnostic.shapelet_explainer, ts)
        
        
        # SHAPELETS FOR THE FAR NEIGHBOR
        far_shapelets = extract_used_shapelets(agnostic_explainers[far_neighbor_index].shapelet_explainer, far_neighbor)
        
        
        far_dist = shapelet_stability_measure(shapelets, far_shapelets)
        print()
        print("far:",far_dist)
        
        # SHAPELETS FOR THE K-NEAREST NEIGHBORS
        near_shapelets = extract_used_shapelets(agnostic_explainers[nearest_neighbor_index].shapelet_explainer, nearest_neighbor)
        dist = shapelet_stability_measure(shapelets, near_shapelets)
        print("near:",dist)
        stability = dist/far_dist
        stabilities.append(stability)
    return np.array(stabilities)
"""

def calculate_shapelet_stability(agnostic_explainers, X_exp_test, n_neighbors = 6):
    X_exp_test = X_exp_test[:,:,0]
    encoder = agnostic_explainers[0].encoder
    decoder = agnostic_explainers[0].decoder
    nbrs = NearestNeighbors(n_neighbors=len(X_exp_test), algorithm='ball_tree').fit(X_exp_test)
    distances, indices = nbrs.kneighbors(X_exp_test)
    stabilities = []
    for i, agnostic in enumerate(agnostic_explainers):
        ts = decoder.predict(agnostic.instance_to_explain_latent.reshape(1,-1)).ravel()
        nearest_neighbors_indices = indices[i][1:n_neighbors]
        
        far_neighbor_index = nearest_neighbors_indices[-1]
        far_neighbor = decoder.predict(encoder.predict(X_exp_test[far_neighbor_index].reshape(1,-1,1))).ravel()
        
        nearest_neighbor_index = nearest_neighbors_indices[0]
        nearest_neighbor = decoder.predict(encoder.predict(X_exp_test[nearest_neighbor_index].reshape(1,-1,1))).ravel()
        
        # SHAPELETS FOR THE INSTANCE TO EXPLAIN
        shapelets = extract_used_shapelets(agnostic.shapelet_explainer, ts)
        
        
        # SHAPELETS FOR THE FAR NEIGHBOR
        far_shapelets = extract_used_shapelets(agnostic_explainers[far_neighbor_index].shapelet_explainer, far_neighbor)
        
        
        far_sim = shapelet_stability_measure(shapelets, far_shapelets)
        #print()
        #print("far:",far_sim)
        
        # SHAPELETS FOR THE K-NEAREST NEIGHBORS
        near_shapelets = extract_used_shapelets(agnostic_explainers[nearest_neighbor_index].shapelet_explainer, nearest_neighbor)
        sim = shapelet_stability_measure(shapelets, near_shapelets)
        #print("near:",sim)
        stability = far_sim/sim
        stabilities.append(stability)
    return np.array(stabilities)  

def calculate_shapelet_multi_stability(agnostic_explainers, X_exp_test):
    X_exp_test = X_exp_test[:,:,0]
    encoder = agnostic_explainers[0].encoder
    decoder = agnostic_explainers[0].decoder
    nbrs = NearestNeighbors(n_neighbors=len(X_exp_test), algorithm='ball_tree').fit(X_exp_test)
    distances, indices = nbrs.kneighbors(X_exp_test)
    stabilities = []
    for i, agnostic in enumerate(agnostic_explainers):
        ts = decoder.predict(agnostic.instance_to_explain_latent.reshape(1,-1)).ravel()
        neighbors_indices = indices[i][1:len(X_exp_test)]
        
        far_neighbor_indices = neighbors_indices[list(range(5,len(X_exp_test)-1,5))]
        #far_neighbor = decoder.predict(encoder.predict(X_exp_test[far_neighbor_index].reshape(1,-1,1))).ravel()
        
        nearest_neighbor_index = neighbors_indices[0]
        nearest_neighbor = decoder.predict(encoder.predict(X_exp_test[nearest_neighbor_index].reshape(1,-1,1))).ravel()
        
        # SHAPELETS FOR THE INSTANCE TO EXPLAIN
        shapelets = extract_used_shapelets(agnostic.shapelet_explainer, ts)
        far_sims = []
        print()
        # SHAPELETS FOR THE FAR NEIGHBOR
        for far_neighbor_index in far_neighbor_indices:
            far_neighbor = decoder.predict(encoder.predict(X_exp_test[far_neighbor_index].reshape(1,-1,1))).ravel()
            far_shapelets = extract_used_shapelets(agnostic_explainers[far_neighbor_index].shapelet_explainer, far_neighbor)
            far_sim = shapelet_stability_measure(shapelets, far_shapelets)
            far_sims.append(far_sim)
            
            print("far:",far_sim)
        
        # SHAPELETS FOR THE K-NEAREST NEIGHBORS
        near_shapelets = extract_used_shapelets(agnostic_explainers[nearest_neighbor_index].shapelet_explainer, nearest_neighbor)
        sim = shapelet_stability_measure(shapelets, near_shapelets)
        print("near:",sim)
        stability = far_sims/sim
        stabilities.append(stability)
    return np.array(stabilities)          
        
        
        
        
        
    

def print_report(results_df, only_global = False):
    global_classification_report = classification_report(results_df["y_blackbox_original_GLOBAL"], results_df["y_surrogate_original_GLOBAL"])
    global_fidelity = accuracy_score(results_df["y_blackbox_original_GLOBAL"], results_df["y_surrogate_original_GLOBAL"])
    local_fidelity = np.nan
    reconstruction_fidelity = np.nan
    if not only_global:
        local_classification_report = classification_report(results_df["y_blackbox_original_LOCAL"], results_df["y_surrogate_reconstructed_LOCAL"])
        reconstrution_classification_report = classification_report(results_df["y_blackbox_original_LOCAL"], results_df["y_blackbox_reconstructed_LOCAL"])
        
        local_fidelity = accuracy_score(results_df["y_blackbox_original_LOCAL"], results_df["y_surrogate_reconstructed_LOCAL"])
        reconstruction_fidelity = accuracy_score(results_df["y_blackbox_original_LOCAL"], results_df["y_blackbox_reconstructed_LOCAL"])
        
    print("reconstruction fidelity: ", reconstruction_fidelity)
    print("local fidelity: ", local_fidelity)
    print("global fidelity: ", global_fidelity)
    
    return({"fid_g": global_fidelity,
            "fid_l": local_fidelity
            })
    
    
        


    
if __name__ == "__main__":
    
    from agnosticglobalexplainer import AgnosticGlobalExplainer, save_shapelet_model, load_shapelet_model
    from myutils import BlackboxPredictWrapper
    from pyts.datasets import make_cylinder_bell_funnel
    from sklearn.model_selection import train_test_split
    from autoencoders import Autoencoder
    from joblib import load, dump
    from blackboxes import build_resnet
    from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, coverage_error
    import pandas as pd
    import time
    import os
    
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
    blackbox_predict = BlackboxPredictWrapper(blackbox, 3)
    encoder = autoencoder.layers[1]
    decoder = autoencoder.layers[2]
    blackbox_input_dimensions = 3
    labels = ["cylinder", "bell", "funnel"]
    
    
    file_path = "./agnostic_explainers/" + dataset_name + "_" + time.strftime("%Y%m%d_%H%M%S")
    os.mkdir(file_path + "/")
    max_iter = 1
    global_surrogate = AgnosticGlobalExplainer(random_state = random_state, max_iter = max_iter, labels = labels)
    global_surrogate.fit(X_exp_train[:,:,0], blackbox_predict.predict(X_exp_train))
    
    agnostic_explainers = build_agnostic_local_explainers(blackbox, 
                                   encoder, 
                                   decoder, 
                                   autoencoder, 
                                   X_exp_test, 
                                   y_exp_test,
                                   blackbox_input_dimensions = blackbox_input_dimensions,
                                   labels = labels,
                                   size = 30,
                                   neigh_type = "geneticp",
                                   ngen = 1,
                                  max_iter=max_iter,
                                  random_state = random_state
                                   )
    
    results_df = get_all_predictions(agnostic_explainers, global_surrogate, blackbox_predict, X_exp_test, blackbox_predict.predict(X_exp_train), encoder, decoder)
    results_df.to_csv(file_path + "/" + "results_df.csv", sep = ";", index = False)
    
    print_report(results_df)
    
    save_shapelet_model(global_surrogate, file_path + "/")
    massive_save_agnostic_local_explainers(agnostic_explainers, file_path, verbose = True)
    
    global_surrogate = load_shapelet_model(file_path + "/")
    agnostic_explainers = massive_load_agnostic_local_explainers(file_path, verbose = True)
    
    results_df_loaded = get_all_predictions(agnostic_explainers, global_surrogate, blackbox_predict, X_exp_test, blackbox_predict.predict(X_exp_train))
    
    print((results_df_loaded.values != results_df.values).sum())
    
    
    
    
