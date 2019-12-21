#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 15:19:29 2019

@author: francesco
"""
from tslearn.shapelets import ShapeletModel, grabocka_params_to_shapelet_size_dict, GlobalMinPooling1D, LocalSquaredDistanceLayer, GlobalArgminPooling1D
from tslearn import shapelets
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import tree
import graphviz
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
#import matplotlib.cm as cm
import pandas as pd
import sys
from joblib import load, dump
import warnings
import keras


def save_shapelet_model(explainer, file_path):
    explainer.shapelet_generator.locator_model_.save(file_path + "_locator.h5")
    explainer.shapelet_generator.model_.save(file_path + "_model.h5")
    explainer.shapelet_generator.transformer_model_.save(file_path + "_transformer.h5")
    explainer.shapelet_generator.locator_model_ = None
    explainer.shapelet_generator.model_ = None
    explainer.shapelet_generator.transformer_model_ = None
    dump(explainer, file_path + "_shapelet_model.pkl")
    
def load_shapelet_model(file_path):
    explainer = load(file_path + "_shapelet_model.pkl")
    explainer.shapelet_generator.locator_model_ = keras.models.load_model(file_path + "_locator.h5", 
                                                                          custom_objects={'GlobalMinPooling1D': GlobalMinPooling1D,
                                                                                          'LocalSquaredDistanceLayer': LocalSquaredDistanceLayer,
                                                                                          'GlobalArgminPooling1D': GlobalArgminPooling1D})
    explainer.shapelet_generator.model_ = keras.models.load_model(file_path + "_model.h5", 
                                                                          custom_objects={'GlobalMinPooling1D': GlobalMinPooling1D,
                                                                                          'LocalSquaredDistanceLayer': LocalSquaredDistanceLayer,
                                                                                          'GlobalArgminPooling1D': GlobalArgminPooling1D})
    explainer.shapelet_generator.transformer_model_ = keras.models.load_model(file_path + "_transformer.h5", 
                                                                          custom_objects={'GlobalMinPooling1D': GlobalMinPooling1D,
                                                                                          'LocalSquaredDistanceLayer': LocalSquaredDistanceLayer,
                                                                                          'GlobalArgminPooling1D': GlobalArgminPooling1D})
    return explainer

def save_reload_shapelet_model(explainer, file_path):
    save_shapelet_model(explainer, file_path)
    explainer = load_shapelet_model(file_path)
    return explainer

def plot_series_shapelet_explanation_old(shapelet_explainer,
                                         ts,
                                         ts_label,
                                         mapper = None,
                                         figsize = (20,3),
                                         color_norm_type = "normal",
                                         vmin = 0,
                                         vmax = 1,
                                         gamma = 2
                                        ):
    sample_id = 0
    dataset = ts.reshape(1,-1)
    dataset_labels = ts_label
    #print("\n",prediction)
    dataset_labels = dataset_labels.ravel()
    dataset_transformed = shapelet_explainer.shapelet_generator.transform(dataset)
    predicted_locations = shapelet_explainer.shapelet_generator.locate(dataset)
    feature = shapelet_explainer.surrogate.tree_.feature
    threshold = shapelet_explainer.surrogate.tree_.threshold
    leave_id = shapelet_explainer.surrogate.apply(dataset_transformed)
    node_indicator = shapelet_explainer.surrogate.decision_path(dataset_transformed)
    node_index = node_indicator.indices[node_indicator.indptr[sample_id]:node_indicator.indptr[sample_id + 1]]
    shapelet_dict = {"shapelet_idxs": [],
                     "threshold_sign": [],
                     "distance": [],
                     "print_out": []
                    }
    
    print('Rules used to predict sample %s: ' % sample_id)
        
    print('sample predicted class: ', dataset_labels[sample_id] if not shapelet_explainer.labels else shapelet_explainer.labels[dataset_labels[sample_id]])
    for node_id in node_index:
        if leave_id[sample_id] == node_id:
            continue

        if (dataset_transformed[sample_id, feature[node_id]] <= threshold[node_id]):
            threshold_sign = "<="
        else:
            threshold_sign = ">"

        print_out = ("decision id node %s : (shapelet n. %s (distance = %s) %s %s)"
              % (node_id,
                 #sample_id,
                 feature[node_id],
                 dataset_transformed[sample_id, feature[node_id]],
                 threshold_sign,
                 threshold[node_id]))
        print(print_out)
        shapelet_dict["print_out"].append(print_out)
        shapelet_dict["shapelet_idxs"].append(feature[node_id])
        shapelet_dict["threshold_sign"].append(threshold_sign)
        shapelet_dict["distance"].append(dataset_transformed[sample_id, feature[node_id]])
        
    test_ts_id = sample_id
    

    
    
    plt.figure(figsize = figsize)
    plt.plot(dataset[test_ts_id].ravel(), c = "gray")
    for i, idx_shp in enumerate(shapelet_dict["shapelet_idxs"]):
        shp = shapelet_explainer.shapelet_generator.shapelets_[idx_shp]
        threshold_sign = shapelet_dict["threshold_sign"][i]
        distance = shapelet_dict["distance"][i]
        distance_color = mapper.to_rgba(distance) if mapper else "r"
        t0 = predicted_locations[test_ts_id, idx_shp]
        plt.plot(np.arange(t0, t0 + len(shp)), shp, 
                 #linewidth=4, 
                 #alpha = 0.5,
                 c = distance_color
                 
                )
    plt.show()
    
    similarity_matrix = []
    threshold_matrix = []
    for i, idx_shp in enumerate(shapelet_dict["shapelet_idxs"]):
        shp = shapelet_explainer.shapelet_generator.shapelets_[idx_shp]
        threshold_sign = shapelet_dict["threshold_sign"][i]
        threshold_val = 0 if threshold_sign == "<=" else 1
        distance = shapelet_dict["distance"][i]
        distance_color = mapper.to_rgba(distance) if mapper else "r"
        t0 = predicted_locations[test_ts_id, idx_shp]
        
        similarity_array = np.full(len(ts), np.NaN)
        similarity_array[t0:t0 + len(shp)] = 1/(1+distance)
        similarity_matrix.append(similarity_array)
        
        threshold_array = np.full(len(ts), np.NaN)
        threshold_array[t0:t0 + len(shp)] = threshold_val
        threshold_matrix.append(threshold_array)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        similarity_mean = np.nanmean(similarity_matrix, axis = 0)
    
    threshold_matrix = np.array(threshold_matrix)
    threshold_aggregated_array = []
    for column_idx in range(threshold_matrix.shape[1]):
        column_values = threshold_matrix[:,column_idx]
        valid_column_values = np.unique(column_values[~np.isnan(column_values)])
        if len(valid_column_values) > 1:
            threshold_aggregated_array.append(2)
        elif len(valid_column_values) == 0:
            threshold_aggregated_array.append(3)
        else:
            threshold_aggregated_array.append(valid_column_values[0])
            
    threshold_aggregated_array = np.array(threshold_aggregated_array)
    similarity_mean_lessthan = np.ma.masked_array(similarity_mean, threshold_aggregated_array != 0)
    similarity_mean_morethan = np.ma.masked_array(similarity_mean, threshold_aggregated_array != 1)
    similarity_mean_mixed = np.ma.masked_array(similarity_mean, threshold_aggregated_array != 2)
    similarity_mean_nan = np.ma.masked_array(np.ones(len(similarity_mean)), threshold_aggregated_array != 3)

    if color_norm_type == "normal":
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax, clip=False)
    elif color_norm_type == "power":
        norm = matplotlib.colors.PowerNorm(vmin=vmin, vmax=vmax, clip=False, gamma = gamma)  
    elif color_norm_type == "log":
        norm = matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax, clip=False)
        
    warm = [[norm(vmin), "lightgrey"], [norm(vmax), "#b40426"]]
    cool = [[norm(vmin), "lightgrey"], [norm(vmax), "#3b4cc0"]] 
    mix = [[norm(vmin), "lightgrey"], [norm(vmax), "#782873"]]
    
    cmap_warm = matplotlib.colors.LinearSegmentedColormap.from_list("", warm)
    cmap_cool = matplotlib.colors.LinearSegmentedColormap.from_list("", cool)
    cmap_mix = matplotlib.colors.LinearSegmentedColormap.from_list("", mix)
    cmap_nan = matplotlib.colors.ListedColormap(["lightgrey"])
    
    #cmap.set_bad(color='lightgray')
    #mapp = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    #colors_list = mapp.to_rgba(norm(similarity_mean))
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(dataset.T, c = "black", alpha = 1)
    ax.pcolorfast((0, len(similarity_mean)-1),
                  ax.get_ylim(),
                  similarity_mean_lessthan[np.newaxis],
                  cmap = cmap_warm, 
                  alpha=1, 
                  vmin = vmin, 
                  vmax = vmax,
                  norm = norm
                  )
    ax.pcolorfast((0, len(similarity_mean)-1),
                  ax.get_ylim(),
                  similarity_mean_morethan[np.newaxis],
                  cmap = cmap_cool, 
                  alpha=1, 
                  vmin = vmin, 
                  vmax = vmax,
                  norm = norm
                  )
    ax.pcolorfast((0, len(similarity_mean)-1),
                  ax.get_ylim(),
                  similarity_mean_mixed[np.newaxis],
                  cmap = cmap_mix, 
                  alpha=1, 
                  vmin = vmin, 
                  vmax = vmax,
                  norm = norm
                  )
    ax.pcolorfast((0, len(similarity_mean)-1),
                  ax.get_ylim(),
                  similarity_mean_nan[np.newaxis],
                  cmap = cmap_nan, 
                  alpha=1, 
                  vmin = vmin, 
                  vmax = vmax,
                  norm = norm
                  )
    #fig.show()
    plt.show()
    
    cmap_cool.set_bad(color='lightgray')
    cmap_warm.set_bad(color='lightgray')
    
    for i, idx_shp in enumerate(shapelet_dict["shapelet_idxs"]):
        
        print(shapelet_dict["print_out"][i])
        plt.figure(figsize = figsize)
        plt.plot(dataset[test_ts_id].ravel(), c = "gray")
        shp = shapelet_explainer.shapelet_generator.shapelets_[idx_shp]
        threshold_sign = shapelet_dict["threshold_sign"][i]
        cmap = cmap_cool if threshold_sign == ">" else cmap_warm
        distance = shapelet_dict["distance"][i]
        distance_color = mapper.to_rgba(distance) if mapper else "r"
        linestyle = "-"
        t0 = predicted_locations[test_ts_id, idx_shp]
        plt.plot(np.arange(t0, t0 + len(shp)), shp, 
                 linewidth=2, 
                 alpha = 0.9,
                 linestyle = linestyle,
                 c = distance_color
                )
        plt.show()
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(dataset.T, c = "black", alpha = 1)
        ax.pcolorfast((0, len(similarity_mean)-1),
                      ax.get_ylim(),
                      similarity_matrix[i][np.newaxis],
                      cmap = cmap, 
                      alpha=1, 
                      vmin = vmin, 
                      vmax = vmax,
                      norm = norm
                      )
        #fig.show()
        plt.show()
        
def plot_series_shapelet_explanation(shapelet_explainer,
                                         ts,
                                         ts_label,
                                         mapper = None,
                                         figsize = (20,3),
                                         color_norm_type = "normal",
                                         vmin = 0,
                                         vmax = 1,
                                         gamma = 2
                                        ):
    sample_id = 0
    dataset = ts.reshape(1,-1)
    dataset_labels = ts_label
    #print("\n",prediction)
    dataset_labels = dataset_labels.ravel()
    dataset_transformed = shapelet_explainer.shapelet_generator.transform(dataset)
    dataset_transformed_binarized = 1*(dataset_transformed < (np.quantile(dataset_transformed,shapelet_explainer.best_quantile)))
    dataset_predicted_labels = shapelet_explainer.predict(dataset)
    predicted_locations = shapelet_explainer.shapelet_generator.locate(dataset)
    feature = shapelet_explainer.surrogate.tree_.feature
    threshold = shapelet_explainer.surrogate.tree_.threshold
    leave_id = shapelet_explainer.surrogate.apply(dataset_transformed_binarized)
    node_indicator = shapelet_explainer.surrogate.decision_path(dataset_transformed_binarized)
    node_index = node_indicator.indices[node_indicator.indptr[sample_id]:node_indicator.indptr[sample_id + 1]]
    shapelet_dict = {"shapelet_idxs": [],
                     "threshold_sign": [],
                     "distance": [],
                     "print_out": []
                    }
    
    print('TREE PATH') 
    print('sample predicted class: ', dataset_predicted_labels[sample_id] if not shapelet_explainer.labels else shapelet_explainer.labels[dataset_predicted_labels[sample_id]])
    print('sample real class: ', dataset_labels[sample_id] if not shapelet_explainer.labels else shapelet_explainer.labels[dataset_labels[sample_id]])
    for node_id in node_index:
        if leave_id[sample_id] == node_id:
            continue

        if (dataset_transformed_binarized[sample_id, feature[node_id]] <= threshold[node_id]):
            threshold_sign = "not-contained"
        else:
            threshold_sign = "contained"
        
        shapelet_dict["shapelet_idxs"].append(feature[node_id])
        shapelet_dict["threshold_sign"].append(threshold_sign)
        shapelet_dict["distance"].append(dataset_transformed[sample_id, feature[node_id]])
        print_out = ("decision id node %s : (shapelet n. %s %s)"
              % (node_id, feature[node_id],threshold_sign,))
        shapelet_dict["print_out"].append(print_out)
        print(print_out)
    #print(shapelet_dict["distance"])
    print()
    print("VERBOSE EXPLANATION")
    print("If", end = " ")
    for i, idx_shp in enumerate(shapelet_dict["shapelet_idxs"]):
        print("shapelet n.", shapelet_dict["shapelet_idxs"][i], "is", shapelet_dict["threshold_sign"][i], end = "")
        if i != len(shapelet_dict["shapelet_idxs"]) - 1:
            print(", and", end = " ")
        else: print(",", end = " ")
    print("then the class is", dataset_predicted_labels[sample_id] if not shapelet_explainer.labels else shapelet_explainer.labels[dataset_predicted_labels[sample_id]])
    
     
    test_ts_id = sample_id
    print()
    print("COMPLETE EXPLANATION")
    print("If", end = " ")
    for i, idx_shp in enumerate(shapelet_dict["shapelet_idxs"]):
        plt.figure(figsize=figsize)#figsize = (figsize[0]/3,figsize[1]/3)
        plt.xlim((0, len(dataset.ravel())-1))
        plt.plot(dataset.T, c = "gray", alpha = 0)
        #plt.axis('equal')
        print("shapelet n.", shapelet_dict["shapelet_idxs"][i], "is", shapelet_dict["threshold_sign"][i], end = "")
        shp = shapelet_explainer.shapelet_generator.shapelets_[idx_shp].ravel()
        
        
        plt.plot(shp, 
                 c = "#b40426" if shapelet_dict["threshold_sign"][i] == "contained" else "#3b4cc0",
                 linewidth=1
                     )
        
        
        plt.axis('off')
        plt.show()
        if i != len(shapelet_dict["shapelet_idxs"]) - 1:
            print("and", end = " ")
        else: print("", end = "")
    print("then the class is", dataset_predicted_labels[sample_id] if not shapelet_explainer.labels else shapelet_explainer.labels[dataset_predicted_labels[sample_id]])
    
    similarity_matrix = []
    threshold_matrix = []
    for i, idx_shp in enumerate(shapelet_dict["shapelet_idxs"]):
        shp = shapelet_explainer.shapelet_generator.shapelets_[idx_shp]
        threshold_sign = shapelet_dict["threshold_sign"][i]
        distance = shapelet_dict["distance"][i]
        t0 = predicted_locations[test_ts_id, idx_shp]
        
        similarity_array = np.full(len(ts), np.NaN)
        similarity_array[t0:t0 + len(shp)] = 1/(1+distance)
        similarity_matrix.append(similarity_array)
        
        threshold_array = np.full(len(ts), np.NaN)
        if threshold_sign == "contained":
            threshold_array[t0:t0 + len(shp)] = 0
        threshold_matrix.append(threshold_array)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        similarity_mean = np.nanmean(similarity_matrix, axis = 0)
    
    threshold_matrix = np.array(threshold_matrix)
    threshold_aggregated_array = []
    for column_idx in range(threshold_matrix.shape[1]):
        column_values = threshold_matrix[:,column_idx]
        valid_column_values = np.unique(column_values[~np.isnan(column_values)])
        if len(valid_column_values) == 0:
            threshold_aggregated_array.append(1)
        else:
            threshold_aggregated_array.append(valid_column_values[0])
            
    threshold_aggregated_array = np.array(threshold_aggregated_array)
    similarity_mean_contained = np.ma.masked_array(similarity_mean, threshold_aggregated_array != 0)
    similarity_mean_nan = np.ma.masked_array(np.ones(len(similarity_mean)), threshold_aggregated_array != 1)

    if color_norm_type == "normal":
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax, clip=False)
    elif color_norm_type == "power":
        norm = matplotlib.colors.PowerNorm(vmin=vmin, vmax=vmax, clip=False, gamma = gamma)  
    elif color_norm_type == "log":
        norm = matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax, clip=False)
    
    cmap_warm = matplotlib.colors.ListedColormap(["#b40426"])
    cmap_nan = matplotlib.colors.ListedColormap(["lightgrey"])
    
    #cmap.set_bad(color='lightgray')
    #mapp = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    #colors_list = mapp.to_rgba(norm(similarity_mean))
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title("Shapelets best alignments")
    ax.plot(dataset.T, c = "gray", alpha = 1)
    for i, idx_shp in enumerate(shapelet_dict["shapelet_idxs"]):
        shp = shapelet_explainer.shapelet_generator.shapelets_[idx_shp]
        threshold_sign = shapelet_dict["threshold_sign"][i]
        distance = shapelet_dict["distance"][i]
        t0 = predicted_locations[test_ts_id, idx_shp]
        ax.plot(np.arange(t0, t0 + len(shp)), shp, 
                 #linewidth=4, 
                 linestyle = "-" if shapelet_dict["threshold_sign"][i] == "contained" else "--",
                 alpha = 1 if shapelet_dict["threshold_sign"][i] == "contained" else 1,
                 label = shapelet_dict["threshold_sign"][i],
                 c = "#b40426" if shapelet_dict["threshold_sign"][i] == "contained" else "#3b4cc0"   
                )
    ax.pcolorfast((0, len(similarity_mean)-1),
                  ax.get_ylim(),
                  similarity_mean_contained[np.newaxis],
                  cmap = cmap_warm, 
                  alpha=0.2, 
                  vmin = vmin, 
                  vmax = vmax,
                  norm = norm
                  )
    """
    ax.pcolorfast((0, len(similarity_mean)-1),
                  ax.get_ylim(),
                  similarity_mean_nan[np.newaxis],
                  cmap = cmap_nan, 
                  alpha=1, 
                  vmin = vmin, 
                  vmax = vmax,
                  norm = norm
                  )
    """
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.show()
    
    

class AgnosticGlobalExplainer(object):
    """Agnostic shapelet tree based explainer"""
    def __init__(self,
                 l = 0.1,
                 r = 2,
                 labels = None,
                 optimizer = "sgd",
                 shapelet_sizes = None,
                 weight_regularizer = .01,
                 max_iter = 100,
                 random_state = None,
                 distance_quantile_threshold = [0.5]
                ):
        
        self.shapelet_generator = None
        self.surrogate = None
        self.labels = labels
        self.l = l
        self.r = r
        self.optimizer = optimizer
        self.shapelet_sizes = shapelet_sizes
        self.weight_regularizer = weight_regularizer
        self.max_iter = max_iter
        self.distance_quantile_threshold = distance_quantile_threshold
        
        self.fidelity = None
        self.graph = None
        self.fitted_transformed_dataset = None
        self.random_state = random_state
        self.fitted_transformed_binarized_dataset = None
        self.best_quantile = None
        
    def fit(self, dataset, dataset_labels):
        n_ts, ts_sz = dataset.shape[:2]
        n_classes = len(set(dataset_labels))
        if self.shapelet_sizes is None:
            shapelet_sizes = grabocka_params_to_shapelet_size_dict(n_ts=n_ts,
                                                           ts_sz=ts_sz,
                                                           n_classes=n_classes,
                                                           l=self.l,
                                                           r=self.r)

            shp_clf = ShapeletModel(n_shapelets_per_size=shapelet_sizes,
                                    optimizer=self.optimizer,
                                    weight_regularizer=self.weight_regularizer,
                                    max_iter=self.max_iter,
                                    random_state = self.random_state,
                                    verbose=0)
        else:
            shp_clf = ShapeletModel(optimizer=self.optimizer,
                                weight_regularizer=self.weight_regularizer,
                                max_iter=self.max_iter,
                                random_state = self.random_state,
                                verbose=0)
        
        shp_clf.fit(dataset, dataset_labels)
        dataset_transformed = shp_clf.transform(dataset)
        self.fitted_transformed_dataset = dataset_transformed
        
        grids = []
        grids_scores = []
        for quantile in self.distance_quantile_threshold:
            transformed_binarized_dataset = 1*(dataset_transformed < (np.quantile(dataset_transformed,quantile)))
            clf = DecisionTreeClassifier()
            param_list = {'min_samples_split': [0.002, 0.01, 0.05, 0.1, 0.2],
                          'min_samples_leaf': [0.001, 0.01, 0.05, 0.1, 0.2],
                          'max_depth': [None, 2, 4, 6, 8, 10, 12, 16]}
            grid = GridSearchCV(clf, param_grid = param_list, scoring='accuracy', n_jobs=-1, verbose = 0)
            grid.fit(transformed_binarized_dataset, dataset_labels)
            grids.append(grid)
            grids_scores.append(grid.best_score_)
        grid = grids[np.argmax(np.array(grids_scores))]
        self.best_quantile = self.distance_quantile_threshold[np.argmax(np.array(grids_scores))]
        self.fitted_transformed_binarized_dataset = 1*(dataset_transformed < (np.quantile(dataset_transformed, self.best_quantile)))
        
        clf = DecisionTreeClassifier(**grid.best_params_)
        clf.fit(self.fitted_transformed_binarized_dataset, dataset_labels)
        
        self.surrogate = clf
        self.shapelet_generator = shp_clf
        self.build_tree_graph()
        
    def predict(self, dataset):
        transformed_dataset = self.shapelet_generator.transform(dataset)
        transformed_binarized_dataset = 1*(transformed_dataset < (np.quantile(self.fitted_transformed_dataset,self.best_quantile)))
        prediction = self.surrogate.predict(transformed_binarized_dataset)
        return prediction
    
    def fit_old(self, dataset, dataset_labels):
        n_ts, ts_sz = dataset.shape[:2]
        n_classes = len(set(dataset_labels))
        if self.shapelet_sizes is None:
            shapelet_sizes = grabocka_params_to_shapelet_size_dict(n_ts=n_ts,
                                                           ts_sz=ts_sz,
                                                           n_classes=n_classes,
                                                           l=self.l,
                                                           r=self.r)

            shp_clf = ShapeletModel(n_shapelets_per_size=shapelet_sizes,
                                    optimizer=self.optimizer,
                                    weight_regularizer=self.weight_regularizer,
                                    max_iter=self.max_iter,
                                    random_state = self.random_state,
                                    verbose=0)
        else:
            shp_clf = ShapeletModel(optimizer=self.optimizer,
                                weight_regularizer=self.weight_regularizer,
                                max_iter=self.max_iter,
                                random_state = self.random_state,
                                verbose=0)
        
        shp_clf.fit(dataset, dataset_labels)
        dataset_transformed = shp_clf.transform(dataset)
        self.fitted_transformed_dataset = dataset_transformed
        clf = DecisionTreeClassifier()
        param_list = {'min_samples_split': [0.002, 0.01, 0.05, 0.1, 0.2],
                      'min_samples_leaf': [0.001, 0.01, 0.05, 0.1, 0.2],
                      'max_depth': [None, 2, 4, 6, 8, 10, 12, 16]}

        grid = GridSearchCV(clf, param_grid = param_list, scoring='accuracy', n_jobs=-1, verbose = 0)
        grid.fit(dataset_transformed, dataset_labels)
        
        clf = DecisionTreeClassifier(**grid.best_params_)
        clf.fit(dataset_transformed, dataset_labels)
        
        self.surrogate = clf
        self.shapelet_generator = shp_clf
        self.build_tree_graph()
        
    
    def predict_old(self, dataset):
        transformed_dataset = self.shapelet_generator.transform(dataset)
        prediction = self.surrogate.predict(transformed_dataset)
        return prediction
    
    def predict_proba_old(self, dataset):
        transformed_dataset = self.shapelet_generator.transform(dataset)
        prediction = self.surrogate.predict_proba(transformed_dataset)
        return prediction
    
    def build_tree_graph(self, out_file=None):
        dot_data = tree.export_graphviz(self.surrogate, out_file=out_file,   
                          class_names=self.labels,  
                          filled=True, rounded=True,  
                          special_characters=True)  
        self.graph = graphviz.Source(dot_data)  
        return self.graph 
    
    def plot_series_shapelet_explanation(self,
                                         ts,
                                         ts_label,
                                         mapper = None,
                                         figsize = (20,3),
                                         color_norm_type = "normal",
                                         vmin = 0,
                                         vmax = 1,
                                         gamma = 2
                                        ):
        # temporary (to delete)
        plot_series_shapelet_explanation(self,
                                         ts,
                                         ts_label,
                                         mapper,
                                         figsize,
                                         color_norm_type,
                                         vmin,
                                         vmax,
                                         gamma)
        
      
    def coverage_score(self, ts):
        ts = ts.reshape(1,-1)
        ts_transformed = self.shapelet_generator.transform(ts)
        ts_transformed_binarized = 1*(ts_transformed < (np.quantile(self.fitted_transformed_dataset,self.best_quantile)))
        ts_leave_id = self.surrogate.apply(ts_transformed_binarized)
        
        all_leaves = self.surrogate.apply(self.fitted_transformed_binarized_dataset)
        coverage = (all_leaves == ts_leave_id[0]).sum()/len(all_leaves)
        
        return coverage
        
    def precision_score(self, ts, y, X = None):
        if X is None:
            X = self.fitted_transformed_binarized_dataset
        else:
            X = self.shapelet_generator.transform(X)
            X = 1*(X < (np.quantile(self.fitted_transformed_dataset,self.best_quantile)))
            
        y_surrogate = self.surrogate.predict(X)
        ts = ts.reshape(1,-1)
        ts_transformed = self.shapelet_generator.transform(ts)
        ts_transformed_binarized = 1*(ts_transformed < (np.quantile(self.fitted_transformed_dataset,self.best_quantile)))
        ts_leave_id = self.surrogate.apply(ts_transformed_binarized)
        
        all_leaves = self.surrogate.apply(X)
        idxs = np.argwhere(all_leaves == ts_leave_id[0])
        
        precision = (y[idxs] == y_surrogate[idxs]).sum()/len(idxs)
        
        return precision
    
        
    
        
        
            


if __name__ == '__main__':
    from pyts.datasets import make_cylinder_bell_funnel
    from sklearn.model_selection import train_test_split
    from joblib import load
    import keras
    from sklearn.metrics import accuracy_score
    
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
    
    
    blackbox = load("./blackbox_checkpoints/cbf_blackbox_knn_20191106_145654.joblib")
    
    #params = {"optimizer":keras.optimizers.Adagrad(lr=.1),"max_iter": 50, "random_state":random_state}
    params = {"optimizer":"sgd","max_iter": 50, "random_state":random_state}#, "distance_quantile_threshold":np.array(list(range(1,10)))/10}
    global_surrogate = AgnosticGlobalExplainer(**params)
    global_surrogate.fit(X_exp_train[:,:,0], blackbox.predict(X_exp_train[:,:,0]))
    print("test fidelity: ", accuracy_score(blackbox.predict(X_exp_test[:,:,0]),
               global_surrogate.predict(X_exp_test[:,:,0])))
    """
    global_surrogate.fit(X_train[:,:,0], blackbox.predict(X_train[:,:,0]))
    global_surrogate.plot_series_shapelet_explanation(X_train[2].ravel(), blackbox.predict(X_train[2].ravel().reshape(1,-1)), figsize=(20,3))
    #plot_series_shapelet_explanation(global_surrogate, X_train[2].ravel(), blackbox.predict(X_train[2].ravel().reshape(1,-1)), figsize=(20,3))
    
    print("test fidelity: ", accuracy_score(blackbox.predict(X_test[:,:,0]),
               global_surrogate.predict(X_test[:,:,0])))
    """
    
    