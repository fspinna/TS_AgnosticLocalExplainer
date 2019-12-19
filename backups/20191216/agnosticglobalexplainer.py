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

"""
def custom_save(explainer, file_path):
    explainer.shapelet_generator.save(file_path + ".h5")
    explainer.shapelet_generator = None
    dump(explainer, file_path + ".pkl")
    
def custom_load(file_path):
    explainer = load(file_path)
    explainer.shapelet_generator = shapelets.load_model(file_path + ".h5")
    return explainer

def custom_save_reload(explainer, file_path):
    custom_save(explainer, file_path)
    custom_load(file_path)
    return explainer
"""

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
                 random_state = None
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
        self.fidelity = None
        self.graph = None
        self.fitted_transformed_dataset = None
        self.random_state = random_state
        
    
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
        
        shp_clf.fit(dataset, dataset_labels)
        dataset_transformed = shp_clf.transform(dataset)
        self.fitted_transformed_dataset = dataset_transformed
        clf = DecisionTreeClassifier()
        param_list = {'min_samples_split': [0.002, 0.01, 0.05, 0.1, 0.2],
                      'min_samples_leaf': [0.001, 0.01, 0.05, 0.1, 0.2],
                      'max_depth': [None, 2, 4, 6, 8, 10, 12, 16]}

        grid = GridSearchCV(clf, param_grid = param_list, scoring='accuracy', n_jobs=-1, verbose = 1)
        grid.fit(dataset_transformed, dataset_labels)
        
        clf = DecisionTreeClassifier(**grid.best_params_)
        clf.fit(dataset_transformed, dataset_labels)
        
        self.surrogate = clf
        self.shapelet_generator = shp_clf
        self.build_tree_graph()
        
    
    def predict(self, dataset):
        transformed_dataset = self.shapelet_generator.transform(dataset)
        prediction = self.surrogate.predict(transformed_dataset)
        return prediction
    
    def predict_proba(self, dataset):
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
        sample_id = 0
        dataset = ts.reshape(1,-1)
        dataset_labels = ts_label
        #print("\n",prediction)
        dataset_labels = dataset_labels.ravel()
        dataset_transformed = self.shapelet_generator.transform(dataset)
        predicted_locations = self.shapelet_generator.locate(dataset)
        feature = self.surrogate.tree_.feature
        threshold = self.surrogate.tree_.threshold
        leave_id = self.surrogate.apply(dataset_transformed)
        node_indicator = self.surrogate.decision_path(dataset_transformed)
        node_index = node_indicator.indices[node_indicator.indptr[sample_id]:node_indicator.indptr[sample_id + 1]]
        shapelet_dict = {"shapelet_idxs": [],
                         "threshold_sign": [],
                         "distance": [],
                         "print_out": []
                        }
        
        print('Rules used to predict sample %s: ' % sample_id)
            
        print('sample predicted class: ', dataset_labels[sample_id] if not self.labels else self.labels[dataset_labels[sample_id]])
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
            shp = self.shapelet_generator.shapelets_[idx_shp]
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
        for i, idx_shp in enumerate(shapelet_dict["shapelet_idxs"]):
            shp = self.shapelet_generator.shapelets_[idx_shp]
            #threshold_sign = shapelet_dict["threshold_sign"][i]
            distance = shapelet_dict["distance"][i]
            distance_color = mapper.to_rgba(distance) if mapper else "r"
            t0 = predicted_locations[test_ts_id, idx_shp]
            similarity_array = np.full(len(ts), np.NaN)
            similarity_array[t0:t0 + len(shp)] = 1/(1+distance)
            similarity_matrix.append(similarity_array)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            similarity_mean = np.nanmean(similarity_matrix, axis = 0)
        if color_norm_type == "normal":
            norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax, clip=False)
        elif color_norm_type == "power":
            norm = matplotlib.colors.PowerNorm(vmin=vmin, vmax=vmax, clip=False, gamma = gamma)  
        elif color_norm_type == "log":
            norm = matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax, clip=False)
        colors = [[norm(vmin), "lightgrey"], [norm(vmax), "#9A1831"]]
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", colors)
        cmap.set_bad(color='lightgray')
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(dataset.T, c = "black", alpha = 1)
        ax.pcolorfast((0, len(similarity_mean)-1),
                      ax.get_ylim(),
                      similarity_mean[np.newaxis],
                      cmap = cmap, 
                      alpha=1, 
                      vmin = vmin, 
                      vmax = vmax,
                      norm = norm
                      )
        #fig.show()
        plt.show()
        
        
        for i, idx_shp in enumerate(shapelet_dict["shapelet_idxs"]):
            
            print(shapelet_dict["print_out"][i])
            plt.figure(figsize = figsize)
            plt.plot(dataset[test_ts_id].ravel(), c = "gray")
            shp = self.shapelet_generator.shapelets_[idx_shp]
            threshold_sign = shapelet_dict["threshold_sign"][i]
            distance = shapelet_dict["distance"][i]
            distance_color = mapper.to_rgba(distance) if mapper else "r"
            if threshold_sign == "<=":
                linestyle = "-."
            else: linestyle = ":"
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
        
    def coverage_score(self, ts):
        ts = ts.reshape(1,-1)
        ts_transformed = self.shapelet_generator.transform(ts)
        ts_leave_id = self.surrogate.apply(ts_transformed)
        
        all_leaves = self.surrogate.apply(self.fitted_transformed_dataset)
        coverage = (all_leaves == ts_leave_id[0]).sum()/len(all_leaves)
        
        return coverage
        
    def precision_score(self, ts, y, X = None):
        if X is None:
            X = self.fitted_transformed_dataset
        else:
            X = self.shapelet_generator.transform(X)
        y_surrogate = self.surrogate.predict(X)
        ts = ts.reshape(1,-1)
        ts_transformed = self.shapelet_generator.transform(ts)
        ts_leave_id = self.surrogate.apply(ts_transformed)
        
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
    params = {"optimizer":"sgd","max_iter": 50, "random_state":random_state}
    global_surrogate = AgnosticGlobalExplainer(**params)
    global_surrogate.fit(X_train[:,:,0], blackbox.predict(X_train[:,:,0]))
    global_surrogate.plot_series_shapelet_explanation(X_train[10].ravel(), blackbox.predict(X_train[10].ravel().reshape(1,-1)), figsize=(18,3))
    
    print("test fidelity: ", accuracy_score(blackbox.predict(X_test[:,:,0]),
               global_surrogate.predict(X_test[:,:,0])))
    
    