#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 16:23:10 2020

@author: francesco
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ruptures as rpt
import shap
import matplotlib.cm as cm
import matplotlib
import sys
from sklearn.neighbors import NearestNeighbors



def segment_ts(ts, model = "rbf", jump = 5, pen = 1, figsize = (20,3), plot = True):
    # detection
    algo = rpt.Pelt(model=model, jump = jump).fit(ts)
    result = algo.predict(pen=pen)

    # display
    if plot:
        rpt.display(ts, true_chg_pts = result, computed_chg_pts=result, figsize = figsize)
        plt.show()
    return result

def generate_segment_list(segmentation):
    # from list of ending segment idxs to list of tuple with starting and ending idxs
    # ex. [5,9,12] --> [(0,5),(5,9),(9,12)]
    segment_list = []
    if len(segmentation) == 1:
        segment_list.append((0,segmentation[0]))
    for i in range(len(segmentation) - 1):
        if i == 0:
            segment_list.append((0, segmentation[i]))
        segment_list.append((segmentation[i], segmentation[i + 1]))
    return segment_list

def gen_val(segment, ts):
    # linear interpolation between two points
    n_points = np.abs(np.diff(segment))[0]
    if segment[1] == len(ts):
        change_amplitude = ts[segment[0]] - ts[segment[1]-1]
    else:
        change_amplitude = ts[segment[0]] - ts[segment[1]]
    steps = abs(change_amplitude/n_points)
    new_vals = []
    for i in range(0,n_points):
        if change_amplitude > 0:
            new_vals.append(ts[segment[0]] - ((i*steps)))
        else:
            new_vals.append(ts[segment[0]] + ((i*steps)))
    return np.array(new_vals)

def linear_consecutive_segmentation(z, segmentation):
    # different type of segmentation: if there are consecutive ones in z the count as only one one
    # ex. z = [0,1,1,0,1,1,1,0] --> z = [0,1,0,1,0]
    new_segmentation = []
    i = 0
    while i < len(segmentation):
        idx = segmentation[i]
        if z[i] == 1:
            if (i + 1 == len(segmentation)) or (z[i + 1] == 0):
                new_segmentation.append(idx)
            else:
                i += 1
                continue
        else:
            new_segmentation.append(idx)
        i += 1
    new_z = z[np.insert(np.diff(z).astype(np.bool), 0, True)]
    return new_z, new_segmentation


def mask_ts(zs, segmentation, ts, background):

    zs = 1 - zs # invert 0 and 1 for np.argwhere
    ts = ts.ravel().copy()

    segment_list = generate_segment_list(segmentation)

    masked_tss = []
    for z in zs:
        if background == "linear_consecutive":
            z, new_segmentation = linear_consecutive_segmentation(z, segmentation)
            segment_list = generate_segment_list(new_segmentation)
        seg_to_change = np.argwhere(z).ravel()
        masked_ts = ts.copy()
        for seg_index in seg_to_change:
            if background in ["linear", "linear_consecutive"]:
                masked_ts[segment_list[seg_index][0]:segment_list[seg_index][1]] = gen_val(segment_list[seg_index], ts)
            else:
                masked_ts[segment_list[seg_index][0]:segment_list[seg_index][1]] = background
        masked_tss.append(masked_ts)
    masked_tss = np.array(masked_tss)
    return masked_tss

def shap_ts(ts, 
            classifier, 
            input_dim = 3, 
            nsamples = 1000, 
            background = "linear", 
            pen = 1,
            model = "rbf",
            jump = 5, plot = True,
            figsize = (20,3)):

    #print(model)
    result = segment_ts(ts, model = model, jump = jump, pen = pen, figsize = figsize, plot = plot)
    def f_3d(z):
        tss = mask_ts(z, result, ts, background)
        tss = tss.reshape(tss.shape[0],tss.shape[1],1)
        return classifier.predict(tss).round()
    def f_2d(z):
        tss = mask_ts(z, result, ts, background)
        return classifier.predict_proba(tss)

    # 2d or 3d classifier input
    if input_dim == 3:
        f = f_3d
    else:
        f = f_2d

    explainer = shap.KernelExplainer(f, data = np.zeros((1,len(result))))

    shap_values = explainer.shap_values(np.ones((1,len(result))), nsamples=nsamples, silent = True)
    #self.shap_output_data.append(self.mask_ts(explainer.synth_data, result, ts, background))
    return shap_values, result



"""
def multi_shap(dataset,
               medoid_idx,
               n = -1, 
               figsize = (20,3), 
               nsamples = 1000,
               background = "linear",
               pen = 1,
               model = "rbf",
               jump = 5,
               ):
    medoid = dataset[medoid_idx]
    if n > len(dataset):
        n = len(dataset)
    if n != -1:
        idxs = np.random.choice(len(dataset), n, replace=False)
        sample_dataset = dataset[idxs]
    else:
        sample_dataset = dataset
    shap_values_array = []
    segmentations = []
    for ts in sample_dataset:
        shap_values, segmentation = shap_ts(ts = ts, 
                    classifier = blackbox, 
                    input_dim = blackbox_input_dimensions,
                    nsamples = nsamples,
                    background = background,
                    pen = pen,
                    model = model,
                    plot = False,
                    jump = jump
                    )
        segmentations.append(segmentation)
        shap_values = np.array(shap_values)
        shap_values = shap_values.reshape(shap_values.shape[0],shap_values.shape[2])
        shap_values_array.append(shap_values)

    plot_aggregated_multi_shap(sample_dataset, shap_values_array, segmentations, figsize = figsize, medoid = medoid)
"""  


def to_colors_by_point(segmentation_list, colors):
    colors_by_point_list = []
    for i, color in enumerate(colors):
        for repetition in range(segmentation_list[i][1]-segmentation_list[i][0]):
            colors_by_point_list.append(color)
    return colors_by_point_list

def shapley_point_by_point(segmentation_list, shapley):
    shapley_by_point_list = []
    for i, shapley in enumerate(shapley):
        for repetition in range(segmentation_list[i][1]-segmentation_list[i][0]):
            shapley_by_point_list.append(shapley)
    return shapley_by_point_list

def shap_output_to_point_by_point(shap_values, segmentation):
    segmentation_list = generate_segment_list(segmentation)
    shap_matrix = [] # classes, shapley values point by point
    for shap_array in shap_values:
        shap_array = shap_array.flatten()
        point_by_point = shapley_point_by_point(segmentation_list, shap_array)
        shap_matrix.append(point_by_point)
    return np.array(shap_matrix)
        
"""
def plot_aggregated_multi_shap(dataset, shap_values_array, segmentations, figsize = (20,3), medoid = None):
    # (batch, classes, 1, segments)
    normalized_shap_values_arrays = []
    segment_lists = [] # (batch, segment_list)
    for i, shap_values in enumerate(shap_values_array):
        normalized_shap_values_array = []

        flat_shap = np.ravel(np.array(shap_values))
        minima = flat_shap.min()
        maxima = flat_shap.max()

        # these are here to avoid error in case there aren't values under or over 0 (for DiverginNorm)
        if minima == 0: minima -= sys.float_info.epsilon
        if maxima == 0: maxima += sys.float_info.epsilon

        norm = matplotlib.colors.DivergingNorm(vmin=minima, vcenter=0, vmax=maxima)

        for shap_array in shap_values:
            normalized_shap_array = norm(shap_array)
            normalized_shap_values_array.append(normalized_shap_array)

        segment_list = generate_segment_list(segmentations[i])
        normalized_shap_values_arrays.append(normalized_shap_values_array)
        segment_lists.append(segment_list)

    colors_by_point_lists = []
    for i, colors_list in enumerate(normalized_shap_values_arrays): 
        colors_by_point_list = []
        for j, colors in enumerate(colors_list):
            colors_by_point = to_colors_by_point(segment_lists[i], colors)
            colors_by_point_list.append(colors_by_point)
        colors_by_point_lists.append(colors_by_point_list)
    #(batch, classes, colors_by_point)
    colors_by_point_array = np.array(colors_by_point_lists)
    #print(colors_by_point_array.shape)
    aggregated_colors = colors_by_point_array.mean(axis = 0)
    #print(aggregated_colors.shape)
    for i in range(aggregated_colors.shape[0]):
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(medoid.reshape(1,-1).T, c = "black", alpha = 1)
        ax.pcolorfast((0, len(aggregated_colors[i,:])-1),
                      ax.get_ylim(),
                      aggregated_colors[i,:][np.newaxis],
                      cmap = "coolwarm", 
                      alpha=1, 
                      vmin = 0, 
                      vmax = 1
                      )
        fig.show()
        plt.show()

"""
"""
def plot_multi_shap(dataset, shap_values_array, segmentations, figsize = (20,3)):
    # (batch, classes, 1, segments)
    colors_lists = []
    segment_lists = []
    for i, shap_values in enumerate(shap_values_array):
        colors_list = []

        flat_shap = np.ravel(np.array(shap_values))
        minima = flat_shap.min()
        maxima = flat_shap.max()

        # these are here to avoid error in case there aren't values under or over 0 (for DiverginNorm)
        if minima == 0: minima -= sys.float_info.epsilon
        if maxima == 0: maxima += sys.float_info.epsilon

        norm = matplotlib.colors.DivergingNorm(vmin=minima, vcenter=0, vmax=maxima)
        mapper = cm.ScalarMappable(norm=norm, cmap=cm.coolwarm)

        for shap_array in shap_values:
            colors = []
            for shap_value in shap_array.ravel():
                colors.append(mapper.to_rgba(shap_value))
            colors_list.append(colors)

        segment_list = generate_segment_list(segmentations[i])
        colors_lists.append(colors_list)
        segment_lists.append(segment_list)

    # for every class
    # for every ts
    # for every segment in ts plot segment
    for j in range(len(shap_values_array[0])): # for every class
        plt.figure(figsize = figsize)
        for k, ts in enumerate(dataset): # for every ts
            segment_list = segment_lists[k]
            colors_list = colors_lists[k]
            for i, segment in enumerate(segment_list): # for every segment in the ts
                seg = pd.Series(ts.ravel())[segment[0]:segment[1]+1]
                if labels:
                    plt.title("Class: " + labels[j])
                else:
                    plt.title("Class: " + str(j))
                plt.plot(seg, c = colors_list[j][i])
        #plt.colorbar(mapper)
        plt.show()
"""

def plot_shap_background(ts, shap_values, segmentation, figsize = (20,3)):
    normalized_shap_values_array = []
    flat_shap = np.ravel(np.array(shap_values))
    minima = flat_shap.min()
    maxima = flat_shap.max()

    # these are here to avoid error in case there aren't values under or over 0 (for DiverginNorm)
    if minima == 0: minima -= sys.float_info.epsilon
    if maxima == 0: maxima += sys.float_info.epsilon

    norm = matplotlib.colors.DivergingNorm(vmin=minima, vcenter=0, vmax=maxima)

    for shap_array in shap_values:
        normalized_shap_array = norm(shap_array).flatten()
        normalized_shap_values_array.append(normalized_shap_array)

    segment_list = generate_segment_list(segmentation)

    colors_by_point_list = []
    for j, colors in enumerate(normalized_shap_values_array):
        colors_by_point = to_colors_by_point(segment_list, colors)
        colors_by_point_list.append(colors_by_point)
    #(batch, classes, colors_by_point)
    colors_by_point_array = np.array(colors_by_point_list)
    #print(colors_by_point_array.shape)
    aggregated_colors = colors_by_point_array#.mean(axis = 0)
    #print(aggregated_colors.shape)
    for i in range(aggregated_colors.shape[0]):
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(ts.reshape(1,-1).T, c = "black", alpha = 1)
        ax.pcolorfast((0, len(aggregated_colors[i,:])-1),
                      ax.get_ylim(),
                      aggregated_colors[i,:][np.newaxis],
                      cmap = "coolwarm", 
                      alpha=1, 
                      vmin = 0, 
                      vmax = 1
                      )
        fig.show()
        plt.show()
       
def plot_shap_background_pbyp(ts, shap_values, figsize = (20,3)):
    
    normalized_shap_values_array = []
    flat_shap = np.ravel(np.array(shap_values))
    minima = flat_shap.min()
    maxima = flat_shap.max()
    
    # these are here to avoid error in case there aren't values under or over 0 (for DiverginNorm)
    if minima == 0: minima -= sys.float_info.epsilon
    if maxima == 0: maxima += sys.float_info.epsilon
    
    norm = matplotlib.colors.DivergingNorm(vmin=minima, vcenter=0, vmax=maxima)
    
    for shap_array in shap_values:
        normalized_shap_array = norm(shap_array).flatten()
        normalized_shap_values_array.append(normalized_shap_array)
    
    colors_by_point_array = np.array(normalized_shap_values_array)
    #print(colors_by_point_array.shape)
    aggregated_colors = colors_by_point_array#.mean(axis = 0)
    #print(aggregated_colors.shape)
    for i in range(aggregated_colors.shape[0]):
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(medoid_ts.reshape(1,-1).T, c = "black", alpha = 1)
        ax.pcolorfast((0, len(aggregated_colors[i,:])-1),
                      ax.get_ylim(),
                      aggregated_colors[i,:][np.newaxis],
                      cmap = "coolwarm", 
                      alpha=1, 
                      vmin = 0, 
                      vmax = 1
                      )
        fig.show()
        plt.show()
        
def shap_f(X, ts_index, classifier, input_dim):
    def f_3d(X):
        return classifier.predict(X[:,:,np.newaxis]).round()
    def f_2d(X):
        return classifier.predict_proba(X)

    # 2d or 3d classifier input
    if input_dim == 3:
        f = f_3d
    else:
        f = f_2d

    explainer = shap.KernelExplainer(f, X)

    shap_values = np.array(explainer.shap_values(X[ts_index].ravel()))
    #self.shap_output_data.append(self.mask_ts(explainer.synth_data, result, ts, background))
    return shap_values
"""
def shap_stability(X_exp_test, 
                   blackbox, 
                   blackbox_input_dimensions, 
                   point_by_point = False, 
                   n_neighbors = 4,
                   quantile = 0.9,
                   **params):
    X_exp_test = X_exp_test[:,:,0]
    nbrs = NearestNeighbors(n_neighbors=len(X_exp_test), algorithm='ball_tree').fit(X_exp_test)
    distances, indices = nbrs.kneighbors(X_exp_test)
    stabilities = []
    for i, ts in enumerate(X_exp_test):
        print()
        #neighborhood_distances = np.array(distances[i][1:n_neighbors]) # the first value is the distance from the point itself (0)
        all_neighborhood_indices = indices[i][1:]
        nearest_neighbors_indices = indices[i][1:n_neighbors]
        
        
        far_neighbor_index = all_neighborhood_indices[np.quantile(range(len(X_exp_test)-1), quantile, interpolation = "lower")]
        far_neighbor = X_exp_test[far_neighbor_index]
        
        # SHAP VALUES FOR THE INSTANCE TO EXPLAIN
        if not point_by_point:
            shap_values, segmentation = shap_ts(ts = ts, 
                                             classifier = blackbox, 
                                             input_dim = blackbox_input_dimensions, 
                                             figsize = (20,3), 
                                             nsamples = params.get("nsamples", 1000),
                                             background = params.get("background", "linear"),
                                             pen = params.get("pen", 1),
                                             model = params.get("peltmodel", "rbf"),
                                             jump = params.get("jump", 5),
                                             plot = False
                                       )
            shap_values = shap_output_to_point_by_point(shap_values, segmentation)
        else:
            shap_values = shap_f(X_exp_test, i, blackbox, blackbox_input_dimensions)
        # SHAP VALUS FOR A DISTANT NEIGHBOR
        if not point_by_point:
            far_shap_values, segmentation = shap_ts(ts = far_neighbor, 
                                             classifier = blackbox, 
                                             input_dim = blackbox_input_dimensions, 
                                             figsize = (20,3), 
                                             nsamples = params.get("nsamples", 1000),
                                             background = params.get("background", "linear"),
                                             pen = params.get("pen", 1),
                                             model = params.get("peltmodel", "rbf"),
                                             jump = params.get("jump", 5),
                                             plot = False
                                       )
            far_shap_values = shap_output_to_point_by_point(far_shap_values, segmentation)
        else:
            far_shap_values = shap_f(X_exp_test, far_neighbor_index, blackbox, blackbox_input_dimensions)
        far_diff = np.abs(shap_values - far_shap_values).mean()
        far_sim = 1 / (1 + far_diff)
        
        print("far:",far_sim)
        
        stability = []
        
        # SHAP VALUS FOR THE K NEAREST NEIGHBORS
        for index in nearest_neighbors_indices:
            neighbor = X_exp_test[index]
            if not point_by_point:
                neighbor_shap_values, segmentation = shap_ts(ts = neighbor, 
                                                 classifier = blackbox, 
                                                 input_dim = blackbox_input_dimensions, 
                                                 nsamples = params.get("nsamples", 1000),
                                                 background = params.get("background", "linear"),
                                                 pen = params.get("pen", 1),
                                                 model = params.get("peltmodel", "rbf"),
                                                 jump = params.get("jump", 5),
                                                 plot = False
                                           )
                neighbor_shap_values = shap_output_to_point_by_point(neighbor_shap_values, segmentation)
            else:
                neighbor_shap_values = shap_f(X_exp_test, index, blackbox, blackbox_input_dimensions)
                
            diff = np.abs(shap_values - neighbor_shap_values).mean()
            sim = 1 / (1 + diff)
            print("near:",sim)
            stability.append(sim)
        stability = np.array(stability)/far_sim
        stabilities.append(np.array([stability.max(), stability.min(), stability.mean()]))
    return np.array(stabilities)
"""
"""
def shap_stability(X_exp_test, 
                   blackbox, 
                   blackbox_input_dimensions, 
                   point_by_point = False, 
                   n_neighbors = 6,
                   **params):
    X_exp_test = X_exp_test[:,:,0]
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(X_exp_test)
    distances, indices = nbrs.kneighbors(X_exp_test)
    stabilities = []
    for i, ts in enumerate(X_exp_test):
        print()
        nearest_neighbors_indices = indices[i][1:n_neighbors]
        
        far_neighbor_index = nearest_neighbors_indices[-1]
        far_neighbor = X_exp_test[far_neighbor_index]
        
        nearest_neighbor_index = nearest_neighbors_indices[0]
        nearest_neighbor = X_exp_test[nearest_neighbor_index]
        
        # SHAP VALUES FOR THE INSTANCE TO EXPLAIN
        if not point_by_point:
            shap_values, segmentation = shap_ts(ts = ts, 
                                             classifier = blackbox, 
                                             input_dim = blackbox_input_dimensions, 
                                             figsize = (20,3), 
                                             nsamples = params.get("nsamples", 1000),
                                             background = params.get("background", "linear"),
                                             pen = params.get("pen", 1),
                                             model = params.get("peltmodel", "rbf"),
                                             jump = params.get("jump", 5),
                                             plot = False
                                       )
            shap_values = shap_output_to_point_by_point(shap_values, segmentation)
        else:
            shap_values = shap_f(X_exp_test, i, blackbox, blackbox_input_dimensions)
        # SHAP VALUS FOR A DISTANT NEIGHBOR
        if not point_by_point:
            far_shap_values, segmentation = shap_ts(ts = far_neighbor, 
                                             classifier = blackbox, 
                                             input_dim = blackbox_input_dimensions, 
                                             figsize = (20,3), 
                                             nsamples = params.get("nsamples", 1000),
                                             background = params.get("background", "linear"),
                                             pen = params.get("pen", 1),
                                             model = params.get("peltmodel", "rbf"),
                                             jump = params.get("jump", 5),
                                             plot = False
                                       )
            far_shap_values = shap_output_to_point_by_point(far_shap_values, segmentation)
        else:
            far_shap_values = shap_f(X_exp_test, far_neighbor_index, blackbox, blackbox_input_dimensions)
        far_diff = np.abs(shap_values - far_shap_values)
        far_diff = np.array([far_diff.mean(), far_diff.max(), far_diff.min()])
        
        print("far:",far_diff)
        
        
        stability = []
        
        # SHAP VALUS FOR THE NEAREST NEIGHBOR
        if not point_by_point:
            neighbor_shap_values, segmentation = shap_ts(ts = nearest_neighbor, 
                                             classifier = blackbox, 
                                             input_dim = blackbox_input_dimensions, 
                                             nsamples = params.get("nsamples", 1000),
                                             background = params.get("background", "linear"),
                                             pen = params.get("pen", 1),
                                             model = params.get("peltmodel", "rbf"),
                                             jump = params.get("jump", 5),
                                             plot = False
                                       )
            neighbor_shap_values = shap_output_to_point_by_point(neighbor_shap_values, segmentation)
        else:
            neighbor_shap_values = shap_f(X_exp_test, nearest_neighbor_index, blackbox, blackbox_input_dimensions)
            
        diff = np.abs(shap_values - neighbor_shap_values)
        diff = np.array([diff.mean(), diff.max(), diff.min()])
        sim = 1 / (1 + diff)
        print("near:",diff)
        stability = diff/far_diff
        stabilities.append(stability)
        
    return np.array(stabilities)
"""

def shap_stability(X_exp_test, 
                   blackbox, 
                   blackbox_input_dimensions, 
                   point_by_point = False, 
                   n_neighbors = 6,
                   **params):
    X_exp_test = X_exp_test[:,:,0]
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(X_exp_test)
    distances, indices = nbrs.kneighbors(X_exp_test)
    stabilities = []
    for i, ts in enumerate(X_exp_test):
        #print()
        nearest_neighbors_indices = indices[i][1:n_neighbors]
        
        far_neighbor_index = nearest_neighbors_indices[-1]
        far_neighbor = X_exp_test[far_neighbor_index]
        
        nearest_neighbor_index = nearest_neighbors_indices[0]
        nearest_neighbor = X_exp_test[nearest_neighbor_index]
        
        # SHAP VALUES FOR THE INSTANCE TO EXPLAIN
        if not point_by_point:
            shap_values, segmentation = shap_ts(ts = ts, 
                                             classifier = blackbox, 
                                             input_dim = blackbox_input_dimensions, 
                                             figsize = (20,3), 
                                             nsamples = params.get("nsamples", 1000),
                                             background = params.get("background", "linear"),
                                             pen = params.get("pen", 1),
                                             model = params.get("peltmodel", "rbf"),
                                             jump = params.get("jump", 5),
                                             plot = False
                                       )
            shap_values = shap_output_to_point_by_point(shap_values, segmentation)
        else:
            shap_values = shap_f(X_exp_test, i, blackbox, blackbox_input_dimensions)
        # SHAP VALUS FOR A DISTANT NEIGHBOR
        if not point_by_point:
            far_shap_values, segmentation = shap_ts(ts = far_neighbor, 
                                             classifier = blackbox, 
                                             input_dim = blackbox_input_dimensions, 
                                             figsize = (20,3), 
                                             nsamples = params.get("nsamples", 1000),
                                             background = params.get("background", "linear"),
                                             pen = params.get("pen", 1),
                                             model = params.get("peltmodel", "rbf"),
                                             jump = params.get("jump", 5),
                                             plot = False
                                       )
            far_shap_values = shap_output_to_point_by_point(far_shap_values, segmentation)
        else:
            far_shap_values = shap_f(X_exp_test, far_neighbor_index, blackbox, blackbox_input_dimensions)
        far_diff = np.abs(shap_values - far_shap_values)
        far_sim = 1 / (1 + far_diff)
        far_sim = np.array([far_sim.mean(), far_sim.max(), far_sim.min()])
        
        #print("far:",far_sim)
        
        
        stability = []
        
        # SHAP VALUS FOR THE NEAREST NEIGHBOR
        if not point_by_point:
            neighbor_shap_values, segmentation = shap_ts(ts = nearest_neighbor, 
                                             classifier = blackbox, 
                                             input_dim = blackbox_input_dimensions, 
                                             nsamples = params.get("nsamples", 1000),
                                             background = params.get("background", "linear"),
                                             pen = params.get("pen", 1),
                                             model = params.get("peltmodel", "rbf"),
                                             jump = params.get("jump", 5),
                                             plot = False
                                       )
            neighbor_shap_values = shap_output_to_point_by_point(neighbor_shap_values, segmentation)
        else:
            neighbor_shap_values = shap_f(X_exp_test, nearest_neighbor_index, blackbox, blackbox_input_dimensions)
            
        diff = np.abs(shap_values - neighbor_shap_values)
        sim = 1 / (1 + diff)
        sim = np.array([sim.mean(), sim.max(), sim.min()])
        #print("near:",sim)
        stability = far_sim/sim
        stabilities.append(stability)
        
    return np.array(stabilities)


def shap_multi_stability(X_exp_test, 
                   blackbox, 
                   blackbox_input_dimensions, 
                   point_by_point = False,
                   **params):
    X_exp_test = X_exp_test[:,:,0]
    nbrs = NearestNeighbors(n_neighbors=len(X_exp_test), algorithm='ball_tree').fit(X_exp_test)
    distances, indices = nbrs.kneighbors(X_exp_test)
    stabilities = []
    for i, ts in enumerate(X_exp_test):
        #print()
        neighbors_indices = indices[i][1:len(X_exp_test)]
        
        far_neighbor_indices = neighbors_indices[list(range(5,len(X_exp_test)-1,5))]
        #far_neighbor = X_exp_test[far_neighbor_index]
        
        nearest_neighbor_index = neighbors_indices[0]
        nearest_neighbor = X_exp_test[nearest_neighbor_index]
        
        # SHAP VALUES FOR THE INSTANCE TO EXPLAIN
        if not point_by_point:
            shap_values, segmentation = shap_ts(ts = ts, 
                                             classifier = blackbox, 
                                             input_dim = blackbox_input_dimensions, 
                                             figsize = (20,3), 
                                             nsamples = params.get("nsamples", 1000),
                                             background = params.get("background", "linear"),
                                             pen = params.get("pen", 1),
                                             model = params.get("peltmodel", "rbf"),
                                             jump = params.get("jump", 5),
                                             plot = False
                                       )
            shap_values = shap_output_to_point_by_point(shap_values, segmentation)
        else:
            shap_values = shap_f(X_exp_test, i, blackbox, blackbox_input_dimensions)
            
        # SHAP VALUS FOR THE NEAREST NEIGHBOR
        if not point_by_point:
            neighbor_shap_values, segmentation = shap_ts(ts = nearest_neighbor, 
                                             classifier = blackbox, 
                                             input_dim = blackbox_input_dimensions, 
                                             figsize = (20,3), 
                                             nsamples = params.get("nsamples", 1000),
                                             background = params.get("background", "linear"),
                                             pen = params.get("pen", 1),
                                             model = params.get("peltmodel", "rbf"),
                                             jump = params.get("jump", 5),
                                             plot = False
                                       )
            neighbor_shap_values = shap_output_to_point_by_point(neighbor_shap_values, segmentation)
        else:
            neighbor_shap_values = shap_f(X_exp_test, nearest_neighbor_index, blackbox, blackbox_input_dimensions)
        diff = np.abs(shap_values - neighbor_shap_values)
        sim = 1 / (1 + diff)
        sim = np.array([sim.mean(), sim.max(), sim.min()])
        
        #print("near:",sim)
        
        
        far_sims = []
        
        # SHAP VALUS FOR THE NEAREST NEIGHBOR
        for far_neighbor_index in far_neighbor_indices:
            far_neighbor = X_exp_test[far_neighbor_index]
            if not point_by_point:
                far_shap_values, segmentation = shap_ts(ts = far_neighbor, 
                                                 classifier = blackbox, 
                                                 input_dim = blackbox_input_dimensions, 
                                                 nsamples = params.get("nsamples", 1000),
                                                 background = params.get("background", "linear"),
                                                 pen = params.get("pen", 1),
                                                 model = params.get("peltmodel", "rbf"),
                                                 jump = params.get("jump", 5),
                                                 plot = False
                                           )
                far_shap_values = shap_output_to_point_by_point(far_shap_values, segmentation)
            else:
                far_shap_values = shap_f(X_exp_test, far_neighbor_index, blackbox, blackbox_input_dimensions)
                
            far_diff = np.abs(shap_values - far_shap_values)
            far_sim = 1 / (1 + far_diff)
            far_sim = np.array([far_sim.mean(), far_sim.max(), far_sim.min()])
            
            far_sims.append(far_sim)
            
        #print("near:",stability)
        stability = far_sims/sim
        #print(stability.shape)
        stabilities.append(stability)
    
    # final shape = (batch, neighbors, metric)
    # for every record a matrix with the n-th neighbor metrics as rows
    return np.array(stabilities)


       
if __name__ == "__main__":
    from pyts.datasets import make_cylinder_bell_funnel
    from sklearn.model_selection import train_test_split
    from joblib import load
    from blackboxes import build_resnet
    
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
    blackbox_input_dimensions = 2
    
    
    blackbox = build_resnet(n_timesteps, n_outputs)
    blackbox.load_weights("./blackbox_checkpoints/cbf_blackbox_resnet_20191106_145242_best_weights_+1.00_.hdf5")
    blackbox_input_dimensions = 3
    
    
    
    
    params = {"background": "linear_consecutive"}
    medoid_ts = X_exp_test[1]
    
    
    #print("blackbox predicted class:", blackbox_predict.predict(medoid_ts.flatten().reshape(1,-1,1)))
    shap_values, segmentation = shap_ts(ts = medoid_ts, 
                                             classifier = blackbox, 
                                             input_dim = blackbox_input_dimensions, 
                                             figsize = (20,3), 
                                             nsamples = params.get("nsamples", 1000),
                                             background = params.get("background", "linear"),
                                             pen = params.get("pen", 1),
                                             model = params.get("peltmodel", "rbf"),
                                             jump = params.get("jump", 5)
                                       )
    
    
    plot_shap_background(ts = medoid_ts, 
                   shap_values = shap_values, 
                   segmentation = segmentation, 
                   figsize = (20,3))
    
    shap_values_pbyp = shap_output_to_point_by_point(shap_values, segmentation)
    
    plot_shap_background_pbyp(ts = medoid_ts, 
                   shap_values = shap_values_pbyp, 
                   figsize = (20,3))
    
    #stabilities = shap_stability(X_exp_test, blackbox, blackbox_input_dimensions, point_by_point = False, n_neighbors = 30, **params)
    stabilities = shap_multi_stability(X_exp_test, blackbox, blackbox_input_dimensions, point_by_point = False, **params)
    """

    shap_values = shap_f(X_exp_test[:,:,0], 1, blackbox, blackbox_input_dimensions)
    plot_shap_background_pbyp(ts = medoid_ts, 
                   shap_values = shap_values_pbyp, 
                   figsize = (20,3))
    """
    