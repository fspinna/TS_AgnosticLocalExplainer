#from lorem import *
#from datamanager import *
#from keras.utils import to_categorical
#from tslearn.datasets import CachedDatasets
#from tslearn.preprocessing import TimeSeriesScalerMinMax
from lore.lorem import LOREM
from lore.util import neuclidean #, record2str, multilabel2str
from lore.datamanager import prepare_dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ruptures as rpt
import keras
import shap
import matplotlib.cm as cm
import matplotlib
import sys
from sklearn.decomposition import PCA #Principal Component Analysis
from scipy.stats import norm
from agnosticglobalexplainer import AgnosticGlobalExplainer
from sklearn.metrics import accuracy_score, pairwise_distances

class AgnosticLocalExplainer(object):
    def __init__(self, 
                 blackbox,
                 encoder, 
                 decoder, 
                 autoencoder, 
                 X_explanation, 
                 y_explanation, 
                 index_to_explain,
                 blackbox_input_dimensions = 3,
                 labels = None
                ):
        """
        # blackbox: a trained blackbox
        # encoder: a trained encoder
        # decoder: a trained decoder
        # autoencoder: a trained autoencoder
        # X_explanation: manifest explanation dataset (not latent) -> 3d shape (n_instances, n_timesteps, n_features)
        # y_explanation: classes of the explanation dataset -> flat 1d array
        # index_to_explain: index of the instance in X_explanation to explain
        # blackbox_input_dimensions: blackbox input type: 2 or 3 dimensions
        # list of labels names
        """
        
        self.blackbox = blackbox
        self.encoder = encoder
        self.decoder = decoder
        self.autoencoder = autoencoder
        self.X_explanation = X_explanation
        self.y_explanation = y_explanation
        self.index_to_explain = index_to_explain
        self.blackbox_input_dimensions = blackbox_input_dimensions
        self.labels = labels
        

        self.Z_latent_instance_neighborhood = None
        self.Z_latent_instance_neighborhood_decoded = None
        self.Zy_latent_instance_neighborhood_labels = None
        
        
        
        self.X_shape = self.X_explanation.shape
        self.X_explanation_latent = self.encoder.predict(self.X_explanation) 
        self.X_shape_latent = self.X_explanation_latent.shape
            
        self.instance_to_explain_latent = self.X_explanation_latent[self.index_to_explain].ravel() 
        self.instance_to_explain = self.X_explanation[self.index_to_explain].ravel() 
        self.instance_to_explain_class = self.y_explanation[self.index_to_explain]
        self.instance_to_explain_blackbox_class = self.blackbox_predict(self.instance_to_explain.reshape(1,-1,1))[0]
        
        
        self.LOREM_Explanation = None

        self.rules_dataframes = None
        self.rules_dataframes_latent = None
        
        self.shap_output_data = None
        
        self.shapelet_explainer = None
        
        self.decoder_count = 0
        
  
    def blackbox_decode_and_predict(self, X):
        # X: 3d array
        # decode the latent space and apply the blackbox
        
        self.decoder_count += 1 # for debug only
        
        decoded = self.decoder.predict(X)
        
        prediction = self.blackbox_predict(decoded)
    
        return prediction
    
    def blackbox_predict(self, X):
        # X: 3d array (batch, timesteps, 1)

        if self.blackbox_input_dimensions == 2:
            X = X.reshape(X.shape[0], X.shape[1]) # 3d to 2d array (batch, timesteps)

        prediction = self.blackbox.predict(X)
    
        if len(prediction.shape) > 1 and (prediction.shape[1] != 1):
            prediction = np.argmax(prediction, axis = 1) # from probability to  predicted class
            
        prediction = prediction.ravel() 
    
        return prediction
    
    def blackbox_predict_proba(self, X):
        # X: 3d array (batch, timesteps, 1)
        if self.blackbox_input_dimensions == 2:
            X = X.reshape(X.shape[0], X.shape[1]) # 3d to 2d array (batch, timesteps)
            prediction = self.blackbox.predict_proba(X)
        else: prediction = self.blackbox.predict(X)
        return prediction
    
    def blackbox_decode_and_predict_proba(self, X):
        # X: 3d array
        # decode the latent space and apply the blackbox
        
        self.decoder_count += 1 # for debug only
        
        decoded = self.decoder.predict(X)
        
        prediction = self.blackbox_predict_proba(decoded)
    
        return prediction
        
    def check_autoencoder_blackbox_consistency(self): 
        # checks if the class of the autoencoded instance is the same as the orginal instance class
        check =  self.instance_to_explain_blackbox_class == (
            self.blackbox_decode_and_predict(self.instance_to_explain_latent.reshape(1,-1))[0])
        print("original class == reconstructed class ---> ", check)
        if check: print("Class: ", 
                        self.instance_to_explain_blackbox_class if not self.labels else self.labels[self.instance_to_explain_blackbox_class] + " ({})".format(self.instance_to_explain_blackbox_class))
        
    def LOREM_neighborhood_generation(self, 
                          neigh_type = 'rndgen', 
                          categorical_use_prob = True,
                          continuous_fun_estimation = False, 
                          size = 1000, 
                          ocr = 0.1, 
                          multi_label=False,
                          one_vs_rest=False,
                          verbose = True, samples = 1000,
                          random_state = 0,
                          ngen = 10):
        
        # generate 2d df of latent space for LOREM method
        columns = [str(i) for i in range(self.X_shape_latent[1])] # attribute names are numbers (timesteps)
        df = pd.DataFrame(self.X_explanation_latent, columns = columns) 
        df["class"] = self.y_explanation.flatten() # should be correct to use y_explanation and not the blackbox prediction (https://github.com/fspinna/LOREM/blob/master/notebooks/test_tabular.ipynb)
        class_name = "class"
        df, feature_names, class_values, numeric_columns, rdf, real_feature_names, features_map = (prepare_dataset(df, class_name))

        X_explanation_latent_2d = self.X_explanation_latent.reshape(self.X_shape_latent[:2]) # 2d latent dataframe

        self.LOREM_explainer = LOREM(K = X_explanation_latent_2d, 
                          bb_predict = self.blackbox_decode_and_predict,
                          bb_predict_proba = self.blackbox_decode_and_predict_proba,
                          feature_names = feature_names, 
                          class_name = class_name, 
                          class_values = class_values, 
                          numeric_columns = numeric_columns, 
                          features_map = features_map,
                          neigh_type = neigh_type, 
                          categorical_use_prob = categorical_use_prob,
                          continuous_fun_estimation = continuous_fun_estimation, 
                          size = size, 
                          ocr = ocr, 
                          multi_label = multi_label, 
                          one_vs_rest = one_vs_rest,
                          random_state = random_state, 
                          verbose = verbose, 
                          ngen = ngen)
        
        samples = size # are these parameters the same?
        
        # neighborhood generation
        self.Z_latent_instance_neighborhood = self.LOREM_explainer.neighgen_fn(self.instance_to_explain_latent, samples)
        
        # generated neighborhood blackbox predicted labels
        self.Zy_latent_instance_neighborhood_labels = self.blackbox_decode_and_predict(self.Z_latent_instance_neighborhood)
        
        if self.LOREM_explainer.multi_label:
            self.Z_latent_instance_neighborhood = np.array([z for z, y in 
                                                            zip(self.Z_latent_instance_neighborhood, 
                                                                self.Zy_latent_instance_neighborhood_labels) 
                                                            if np.sum(y) > 0])
            self.Zy_latent_instance_neighborhood_labels = self.blackbox_decode_and_predict(
                self.Z_latent_instance_neighborhood)
        
        if self.LOREM_explainer.verbose:
            if not self.LOREM_explainer.multi_label:
                neigh_class, neigh_counts = np.unique(self.Zy_latent_instance_neighborhood_labels, return_counts=True)
                neigh_class_counts = {class_values[k]: v for k, v in zip(neigh_class, neigh_counts)}
            else:
                neigh_counts = np.sum(self.Zy_latent_instance_neighborhood_labels, axis=0)
                neigh_class_counts = {class_values[k]: v for k, v in enumerate(neigh_counts)}

            print('synthetic neighborhood class counts %s' % neigh_class_counts)
            
    def print_rules_n(self):
        for rule in self.rules_dataframes.keys():
            print(rule + ": " + str(len(self.rules_dataframes[rule]["df"])) + " time series")
            
    def LOREM_rules_random_augmentation(self, 
                                        rules_to_augment = None,
                                        categorical_use_prob = True,
                                        continuous_fun_estimation = False, 
                                        size = 1000, 
                                        ocr = 0.1, 
                                        multi_label=False,
                                        one_vs_rest=False,
                                        verbose = True, 
                                        samples = 1000,
                                        random_state = 0,
                                        ngen = 10):
        
                               
    
        """
        neighgen = RandomGenerator(self.bb_predict, self.feature_values, self.features_map, nbr_features,
                                       nbr_real_features, numeric_columns_index, ocr=ocr)
        """
        
        columns = [str(i) for i in range(self.X_shape_latent[1])] # attribute names are numbers (timesteps)
        df = pd.DataFrame(self.X_explanation_latent, columns = columns) 
        df["class"] = self.y_explanation.flatten() 
        class_name = "class"
        df, feature_names, class_values, numeric_columns, rdf, real_feature_names, features_map = (prepare_dataset(df, class_name))

        X_explanation_latent_2d = self.X_explanation_latent # 2d latent dataframe
        
        LOREM_explainer = LOREM(K = X_explanation_latent_2d, 
                          bb_predict = self.blackbox_decode_and_predict, 
                          feature_names = feature_names, 
                          class_name = class_name, 
                          class_values = class_values, 
                          numeric_columns = numeric_columns, 
                          features_map = features_map,
                          neigh_type = "random", 
                          categorical_use_prob = categorical_use_prob,
                          continuous_fun_estimation = continuous_fun_estimation, 
                          size = size, 
                          ocr = ocr, 
                          multi_label = multi_label, 
                          one_vs_rest = one_vs_rest,
                          random_state = random_state, 
                          verbose = verbose, 
                          ngen = ngen)
        
        samples = size # are these parameters the same?
        
        # neighborhood generation
        Z_latent_instance_neighborhood = LOREM_explainer.neighgen_fn(self.instance_to_explain_latent, samples)
        
        # generated neighborhood blackbox predicted labels
        Zy_latent_instance_neighborhood_labels = self.blackbox_decode_and_predict(Z_latent_instance_neighborhood)
        
        if LOREM_explainer.multi_label:
            Z_latent_instance_neighborhood = np.array([z for z, y in 
                                                            zip(Z_latent_instance_neighborhood, 
                                                                Zy_latent_instance_neighborhood_labels) 
                                                            if np.sum(y) > 0])
            Zy_latent_instance_neighborhood_labels = self.blackbox_decode_and_predict(
                Z_latent_instance_neighborhood)
        
        if LOREM_explainer.verbose:
            if not LOREM_explainer.multi_label:
                neigh_class, neigh_counts = np.unique(Zy_latent_instance_neighborhood_labels, return_counts=True)
                neigh_class_counts = {class_values[k]: v for k, v in zip(neigh_class, neigh_counts)}
            else:
                neigh_counts = np.sum(Zy_latent_instance_neighborhood_labels, axis=0)
                neigh_class_counts = {class_values[k]: v for k, v in enumerate(neigh_counts)}

            print('synthetic neighborhood class counts %s' % neigh_class_counts)
        
        Z_latent_instance_neighborhood_decoded = self.decoder.predict(Z_latent_instance_neighborhood)[:,:,0]
        
        if not rules_to_augment:
            rules_to_augment = self.rules_dataframes.keys()
        for rule in rules_to_augment:
            count = 0
            for i, latent_instance in enumerate(Z_latent_instance_neighborhood):
                if self.ABELE_is_covered(self.rules_dataframes[rule]["Rule_obj"], latent_instance):
                    decoded_instance = Z_latent_instance_neighborhood_decoded[i]
                    self.rules_dataframes[rule]["df"] = np.vstack([self.rules_dataframes[rule]["df"], decoded_instance])
                    self.rules_dataframes_latent[rule]["df"] = np.vstack([self.rules_dataframes_latent[rule]["df"], latent_instance])
                    count += 1
            print(rule + " augmented by " + str(count) + " random instances (" + str(len(self.rules_dataframes[rule]["df"])) + ") time series total")
        
        print("recomputing medoids...")
        for rule in rules_to_augment: 
            distance_matrix = pairwise_distances(self.rules_dataframes_latent[rule]["df"], n_jobs = -1)
            medoid_idx = np.argmin(distance_matrix.sum(axis=0))
            self.rules_dataframes_latent[rule]["medoid_idx"] = medoid_idx
            self.rules_dataframes[rule]["medoid_idx"] = medoid_idx
    
    
    
    
    def generate_premises_by_attribute(self, LOREM_Rule):
        premises_by_att = dict()
        for premise in LOREM_Rule.premises:
            if int(premise.att) in premises_by_att:
                premises_by_att[int(premise.att)].append(premise)
            else: 
                premises_by_att[int(premise.att)] = [premise]
        return premises_by_att
            
    def parse_LOREM_Condition(self, vector, LOREM_Condition):
        if LOREM_Condition.op == "<=":
            return vector <= LOREM_Condition.thr
        else:
            return vector >= LOREM_Condition.thr
    
    def generate_bounded_instance(self, premises_by_att):
        z = np.zeros(len(self.LOREM_explainer.feature_values))
        for i in range(len(z)):
            if i in premises_by_att:
                conditions_truth_array = np.ones(len(self.LOREM_explainer.feature_values[i]))
                for condition in premises_by_att[i]:
                    condition_truth_array = self.parse_LOREM_Condition(self.LOREM_explainer.feature_values[i], condition)
                    conditions_truth_array = conditions_truth_array * condition_truth_array
                idxs_to_choose = np.argwhere(conditions_truth_array)
                values_to_choose = self.LOREM_explainer.feature_values[i][idxs_to_choose].ravel()
                z[i] = np.random.choice(values_to_choose, size=1, replace=True)
            else:
                z[i] = np.random.choice(self.LOREM_explainer.feature_values[i], size=1, replace=True)
        return z
    
    def check_bounded_instance_generation(self, Z, rule_key):
        for z in Z:
            if not self.ABELE_is_covered(self.rules_dataframes[rule_key]["Rule_obj"], z):
                raise Exception("bounded instance wrongly generated")
            
    def rule_random_augmentation(self, rule_key, num_samples = 100):
        premises_by_att = self.generate_premises_by_attribute(self.rules_dataframes[rule_key]["Rule_obj"])
        Z = np.zeros((num_samples, len(self.LOREM_explainer.feature_values)))
        for j in range(num_samples):
            Z[j] = self.generate_bounded_instance(premises_by_att)
        self.check_bounded_instance_generation(Z, rule_key)
        Z_decoded = self.decoder.predict(Z)[:,:,0]
        self.rules_dataframes[rule_key]["df"] = np.append(self.rules_dataframes[rule_key]["df"], Z_decoded, axis = 0)
        self.rules_dataframes_latent[rule_key]["df"] = np.append(self.rules_dataframes_latent[rule_key]["df"], Z, axis = 0)
        
        print("Recomputing medoid... ", end = "")
        distance_matrix = pairwise_distances(self.rules_dataframes_latent[rule_key]["df"], n_jobs = -1)
        medoid_idx = np.argmin(distance_matrix.sum(axis=0))
        self.rules_dataframes_latent[rule_key]["medoid_idx"] = medoid_idx
        self.rules_dataframes[rule_key]["medoid_idx"] = medoid_idx
        print("Done!")
    
    def rules_random_augmentation(self, rules_to_augment = None, num_samples = 100):
        if rules_to_augment is None: rules_to_augment = self.rules_dataframes.keys()
        for rule in rules_to_augment:
            self.rule_random_augmentation(rule, num_samples)
        self.print_rules_n()
        
    def rules_balance_augmentation(self, plus = 0):
        max_len = -1
        for rule in self.rules_dataframes.keys():
            if len(self.rules_dataframes[rule]["df"]) > max_len:
                max_len = len(self.rules_dataframes[rule]["df"])
        max_len += plus
        for rule in self.rules_dataframes.keys():
            if max_len - len(self.rules_dataframes[rule]["df"]) > 0:
                self.rule_random_augmentation(rule, max_len - len(self.rules_dataframes[rule]["df"]))
        self.print_rules_n()
            
            
            
        
    """ 
    #
     def generate_synthetic_instance(self, from_z=None, mutpb=1.0):
        z = np.zeros(self.nbr_features) if from_z is None else from_z
        for i in range(self.nbr_real_features):
            if np.random.random() <= mutpb:
                real_feature_value = np.random.choice(self.feature_values[i], size=1, replace=True)
                if i in self.numeric_columns_index:
                    z[i] = real_feature_value
                else:
                    idx = self.features_map[i][real_feature_value[0]]
                    z[idx] = 1.0
        return z
        
    def generate(self, x, num_samples=1000):
        Z = np.zeros((num_samples, self.nbr_features))
        for j in range(num_samples):
            Z[j] = self.generate_synthetic_instance()
    """
        
        
    def LOREM_weights_calculation(self, use_weights = True, metric = neuclidean):
        if not use_weights:
            weights = None 
        else: 
            weights = self.LOREM_explainer.__calculate_weights__(self.Z_latent_instance_neighborhood, metric)
        return weights

    def LOREM_tree_rules_extraction(self):
        weights = self.LOREM_weights_calculation(use_weights = True, metric = neuclidean)
        if self.LOREM_explainer.one_vs_rest and self.LOREM_explainer.multi_label:
            exp = self.LOREM_explainer._LOREM__explain_tabular_instance_multiple_tree(
                self.instance_to_explain_latent, 
                self.Z_latent_instance_neighborhood, 
                self.Zy_latent_instance_neighborhood_labels, 
                weights)
        else:  # binary, multiclass, multilabel all together
            exp = self.LOREM_explainer._LOREM__explain_tabular_instance_single_tree(
                self.instance_to_explain_latent, 
                self.Z_latent_instance_neighborhood, 
                self.Zy_latent_instance_neighborhood_labels, 
                weights)
        self.LOREM_Explanation = exp
    
    def ABELE_is_covered(self, LOREM_Rule, latent_instance):
        # checks if a latent instance satisfy a LOREM_Rule
        xd = self.ABELE_vector2dict(latent_instance, self.LOREM_explainer.feature_names)
        for p in LOREM_Rule.premises:
            if p.op == '<=' and xd[p.att] > p.thr:
                return False
            elif p.op == '>' and xd[p.att] <= p.thr:
                return False
        return True
    
    def ABELE_vector2dict(self, x, feature_names):
        return {k: v for k, v in zip(feature_names, x)}
    
    
    def build_rules_dataframes(self):
        
        # decodes the latent neighborhood
        self.Z_latent_instance_neighborhood_decoded = self.decoder.predict(self.Z_latent_instance_neighborhood)[:,:,0]
        
        # creates a dictionary having as keys ["rule", "crule0", ... , "cruleN"]
        # and as values a dictionary with keys ["Rule_obj", "df"]
        rules_dataframes = dict()
        rules_dataframes["rule"] = {"Rule_obj": self.LOREM_Explanation.rule, "df": []}
        
        rules_dataframes_latent = dict()
        rules_dataframes_latent["rule"] = {"Rule_obj": self.LOREM_Explanation.rule, "df": []}
        
        
        for i, counterfactual in enumerate(self.LOREM_Explanation.crules):
            rules_dataframes["crule" + str(i)] = {"Rule_obj": counterfactual, "df": []}
            rules_dataframes_latent["crule" + str(i)] = {"Rule_obj": counterfactual, "df": []}
        print("N.RULES = ", 1) 
        print("N.COUNTERFACTUAL = ", len(self.LOREM_Explanation.crules))
        
        for i, latent_instance in enumerate(self.Z_latent_instance_neighborhood):
            for rule in rules_dataframes.keys():
                if self.ABELE_is_covered(rules_dataframes[rule]["Rule_obj"], latent_instance):
                    decoded_instance = self.Z_latent_instance_neighborhood_decoded[i]
                    rules_dataframes[rule]["df"].append(decoded_instance)
                    rules_dataframes_latent[rule]["df"].append(latent_instance)
        
        for rule in rules_dataframes.keys(): 
            rules_dataframes[rule]["df"] = pd.DataFrame(rules_dataframes[rule]["df"]).values
            rules_dataframes_latent[rule]["df"] = pd.DataFrame(rules_dataframes_latent[rule]["df"]).values
            
            distance_matrix = pairwise_distances(rules_dataframes_latent[rule]["df"], n_jobs = -1)
            medoid_idx = np.argmin(distance_matrix.sum(axis=0))
            rules_dataframes_latent[rule]["medoid_idx"] = medoid_idx
            rules_dataframes[rule]["medoid_idx"] = medoid_idx
            print(rule + ": " + str(len(rules_dataframes[rule]["df"])) + " time series")
            
        self.rules_dataframes = rules_dataframes
        self.rules_dataframes_latent = rules_dataframes_latent
        
    
    def plot_rules_dataframes(self, figsize=(20,8)):
        colors = ["b", "g", "c", "m", "k", "orange", "olive", "pink"]
        for rule in self.rules_dataframes.keys():
            plt.figure(figsize=figsize)
            #plt.suptitle(rule + " - " + str(self.rules_dataframes[rule]["df"].shape[0]) +  " time series") 
            plt.title(rule + ": " + str(self.rules_dataframes[rule]["Rule_obj"]) + " - " + str(self.rules_dataframes[rule]["df"].shape[0]) +  " time series")
            for ts in self.rules_dataframes[rule]["df"]:
                plt.plot(ts, c = "red", alpha = 0.5)
            plt.plot(self.rules_dataframes[rule]["df"][self.rules_dataframes[rule]["medoid_idx"]], c = "black", linestyle='--')
            #plt.plot(self.rules_dataframes[rule]["df"].mean(axis = 0), c = "black", linestyle='--')
            plt.show()
        plt.figure(figsize=figsize)
        
        plt.title("Rules Medoids")
        for i, rule in enumerate(self.rules_dataframes.keys()):
            plt.plot(self.rules_dataframes[rule]["df"][self.rules_dataframes[rule]["medoid_idx"]], c = colors[i%len(colors)], label = rule)
        plt.legend()
        plt.show()
        """
        plt.title("Rule Averages")
        for i, rule in enumerate(self.rules_dataframes.keys()):
            plt.plot(self.rules_dataframes[rule]["df"].mean(axis = 0), c = colors[i%len(colors)], label = rule)
        plt.legend()
        plt.show()
        """
        """
        plt.figure(figsize=figsize)
        plt.title("Rule Medians")
        for i, rule in enumerate(self.rules_dataframes.keys()):
            plt.plot(np.median(self.rules_dataframes[rule]["df"], axis = 0), c = colors[i%len(colors)], label = rule)
        plt.legend()
        plt.show()
        """
        
    def plot_rules_heatmaps(self, figsize=(20,4)):
        for rule in self.rules_dataframes.keys():
            fig = plt.figure(figsize = figsize)
            ax = fig.add_subplot(111)
            ax.matshow(self.rules_dataframes[rule]["df"], interpolation=None, aspect='auto', cmap = "viridis")
            ax.set_title(rule + ": " + str(self.rules_dataframes[rule]["Rule_obj"]) + " - " + str(self.rules_dataframes[rule]["df"].shape[0]) +  " time series")
            plt.show()
        
        fig = plt.figure(figsize = figsize)
        mean_df = []
        for rule in self.rules_dataframes.keys():
            mean_df.append(self.rules_dataframes[rule]["df"].mean(axis = 0))
        mean_df = pd.DataFrame(mean_df)
        ax = fig.add_subplot(111)
        ax.matshow(mean_df, interpolation=None, aspect='auto', cmap = "viridis")
        ax.set_title("Rule Averages")
        plt.show()
        
    def segment_ts(self, ts, model = "rbf", jump = 5, pen = 1, figsize = (20,3), plot = True):
        # detection
        algo = rpt.Pelt(model=model, jump = jump).fit(ts)
        result = algo.predict(pen=pen)
    
        # display
        if plot:
            rpt.display(ts, true_chg_pts = result, computed_chg_pts=result, figsize = figsize)
            plt.show()
        return result
    
    def generate_segment_list(self, segmentation):
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
    
    def gen_val(self, segment, ts):
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
    
    def linear_consecutive_segmentation(self, z, segmentation):
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
    
    
    def mask_ts(self, zs, segmentation, ts, background):
        
        zs = 1 - zs # invert 0 and 1 for np.argwhere
        ts = ts.ravel().copy()
        
        segment_list = self.generate_segment_list(segmentation)
        
        masked_tss = []
        for z in zs:
            if background == "linear_consecutive":
                z, new_segmentation = self.linear_consecutive_segmentation(z, segmentation)
                segment_list = self.generate_segment_list(new_segmentation)
            seg_to_change = np.argwhere(z).ravel()
            masked_ts = ts.copy()
            for seg_index in seg_to_change:
                if background in ["linear", "linear_consecutive"]:
                    masked_ts[segment_list[seg_index][0]:segment_list[seg_index][1]] = self.gen_val(segment_list[seg_index], ts)
                else:
                    masked_ts[segment_list[seg_index][0]:segment_list[seg_index][1]] = background
            masked_tss.append(masked_ts)
        masked_tss = np.array(masked_tss)
        return masked_tss
    
    
    def plot_shap(self, ts, shap_values, segmentation, figsize = (20,3)):
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
    
        segment_list = self.generate_segment_list(segmentation)
        
        for j in range(len(shap_values)):
            plt.figure(figsize = figsize)
            for i, segment in enumerate(segment_list):
                seg = pd.Series(ts.ravel())[segment[0]:segment[1]+1]
                if self.labels:
                    plt.title("Class: " + self.labels[j])
                else:
                    plt.title("Class: " + str(j))
                plt.plot(seg, c = colors_list[j][i])
            plt.colorbar(mapper)
            plt.show()
            
    def plot_shap_by_class(self, rules_tss, rules_shap_values, rules_segmentations, figsize = (20,3)):
        colors_lists = []
        segment_lists = []
        for i, shap_values in enumerate(rules_shap_values):
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
        
            segment_list = self.generate_segment_list(rules_segmentations[i])
            colors_lists.append(colors_list)
            segment_lists.append(segment_list)
        
        for j in range(len(rules_shap_values[0])): # for every class
            for k, ts in enumerate(rules_tss): # for every ts
                segment_list = segment_lists[k]
                colors_list = colors_lists[k]
                plt.figure(figsize = figsize)
                for i, segment in enumerate(segment_list): # for every segment in the ts
                    seg = pd.Series(ts.ravel())[segment[0]:segment[1]+1]
                    if self.labels:
                        plt.title("Class: " + self.labels[j])
                    else:
                        plt.title("Class: " + str(j))
                    plt.plot(seg, c = colors_list[j][i])
                plt.colorbar(mapper)
                plt.show()
            
    def shap_ts(self, 
                ts, 
                classifier, 
                input_dim = 3, 
                nsamples = 1000, 
                background = "linear", 
                pen = 1,
                model = "rbf",
                jump = 5, plot = True,
                figsize = (20,3)):
        
        #print(model)
        result = self.segment_ts(ts, model = model, jump = jump, pen = pen, figsize = figsize, plot = plot)
        def f_3d(z):
            tss = self.mask_ts(z, result, ts, background)
            tss = tss.reshape(tss.shape[0],tss.shape[1],1)
            return classifier.predict(tss).round()
            #return to_categorical(np.argmax(blackbox.predict(tss),axis = 1))
            #return blackbox.predict(tss)
            #return np.argmax(blackbox.predict(tss),axis = 1)
        def f_2d(z):
            tss = self.mask_ts(z, result, ts, background)
            return classifier.predict_proba(tss)
            #return to_categorical(np.argmax(blackbox.predict(tss),axis = 1))
            #return blackbox.predict(tss)
            #return np.argmax(blackbox.predict(tss),axis = 1)
            
        # 2d or 3d classifier input
        if input_dim == 3:
            f = f_3d
        else:
            f = f_2d
        
        explainer = shap.KernelExplainer(f, data = np.zeros((1,len(result))))
        
        shap_values = explainer.shap_values(np.ones((1,len(result))), nsamples=nsamples)
        #self.shap_output_data.append(self.mask_ts(explainer.synth_data, result, ts, background))
        return shap_values, result
    
    
    def multi_shap(self, 
                   dataset,
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
            shap_values, segmentation = self.shap_ts(ts = ts, 
                        classifier = self.blackbox, 
                        input_dim = self.blackbox_input_dimensions,
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
            
        self.plot_multi_shap(sample_dataset, shap_values_array, segmentations, figsize = figsize)
        self.plot_aggregated_multi_shap(sample_dataset, shap_values_array, segmentations, figsize = figsize, medoid = medoid)
        #return shap_values_array
        max_len = 0
        for shap_values in shap_values_array:
            if shap_values.shape[1] > max_len:
                max_len = shap_values.shape[1]
                
        for i, shap_values in enumerate(shap_values_array):
            if shap_values.shape[1] < max_len:
                padding = max_len - shap_values.shape[1]
                shap_values_array[i] = np.pad(shap_values_array[i], ((0, 0), (0, padding)), 'constant', constant_values = 0)
        shap_values_array = np.array(shap_values_array)
        
        
        shap_values_array = np.transpose(shap_values_array, (1, 0, 2))
        self.shap_heatmap(shap_values_array, figsize = (8,8))
        
    
    def to_colors_by_point(self, segmentation_list, colors):
        colors_by_point_list = []
        for i, color in enumerate(colors):
            for repetition in range(segmentation_list[i][1]-segmentation_list[i][0]):
                colors_by_point_list.append(color)
        return colors_by_point_list
    
    def plot_aggregated_multi_shap(self, dataset, shap_values_array, segmentations, figsize = (20,3), medoid = None):
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
        
            segment_list = self.generate_segment_list(segmentations[i])
            normalized_shap_values_arrays.append(normalized_shap_values_array)
            segment_lists.append(segment_list)
          
        colors_by_point_lists = []
        for i, colors_list in enumerate(normalized_shap_values_arrays): 
            colors_by_point_list = []
            for j, colors in enumerate(colors_list):
                colors_by_point = self.to_colors_by_point(segment_lists[i], colors)
                colors_by_point_list.append(colors_by_point)
            colors_by_point_lists.append(colors_by_point_list)
        #(batch, classes, colors_by_point)
        colors_by_point_array = np.array(colors_by_point_lists)
        #print(colors_by_point_array.shape)
        aggregated_colors = colors_by_point_array.mean(axis = 0)
        #print(aggregated_colors.shape)
        
        for i in range(aggregated_colors.shape[0]):
            """
            if not medoid_idx:
                ax = pd.DataFrame(dataset.T).plot(c = "black", legend=False, figsize=(20,3), alpha = 0.8)
            """
            ax = pd.DataFrame(medoid.reshape(1,-1).T).plot(c = "black", legend=False, figsize=(20,3), alpha = 1)
            ax.set_title("Class: " + self.labels[i] if self.labels else "Class: " + str(i))
            ax.pcolorfast(ax.get_xlim(), 
                          ax.get_ylim(),
                          aggregated_colors[i,:][np.newaxis],
                          cmap = "coolwarm", alpha=1, 
                          vmin = 0, 
                          vmax = 1
                          )
      
        
    def plot_multi_shap(self, dataset, shap_values_array, segmentations, figsize = (20,3)):
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
        
            segment_list = self.generate_segment_list(segmentations[i])
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
                    if self.labels:
                        plt.title("Class: " + self.labels[j])
                    else:
                        plt.title("Class: " + str(j))
                    plt.plot(seg, c = colors_list[j][i])
            #plt.colorbar(mapper)
            plt.show()
    
    def shap_heatmap(self, shap_values_array, figsize = (8,8)):
        # shap_values_array -> 3d: (classes, batch, segments)
        minima = shap_values_array.min()
        maxima = shap_values_array.max()
        
        # these are here to avoid error in case there aren't values under or over 0 (for DiverginNorm)
        if minima == 0: minima -= sys.float_info.epsilon
        if maxima == 0: maxima += sys.float_info.epsilon
    
        norm = matplotlib.colors.DivergingNorm(vmin=minima, vcenter=0, vmax=maxima)
        for i, class_array in enumerate(shap_values_array):
            fig = plt.figure(figsize = figsize)
            ax = fig.add_subplot(111)
            ax.set_title("class: " + str(i) if not self.labels else "class: " + str(self.labels[i]) + " ({})".format(str(i)))
            ax.matshow(class_array, norm = norm, cmap = "coolwarm", aspect="auto")
            plt.show()
            
    def build_shapelet_explainer(self, l=0.1, r=2, weight_regularizer=.01, optimizer="sgd", max_iter=100, random_state = None):
        self.shapelet_explainer = AgnosticGlobalExplainer(labels = self.labels,
                                                          l = l,
                                                          r = r, 
                                                          weight_regularizer = weight_regularizer,
                                                          optimizer = optimizer,
                                                          random_state = random_state,
                                                          max_iter = max_iter) 
        if self.Z_latent_instance_neighborhood_decoded is None:
            Z_latent_instance_neighborhood_decoded = self.decoder.predict(self.Z_latent_instance_neighborhood)[:,:,0]
        else:
            Z_latent_instance_neighborhood_decoded = self.Z_latent_instance_neighborhood_decoded
        self.shapelet_explainer.fit(Z_latent_instance_neighborhood_decoded,
                                    self.Zy_latent_instance_neighborhood_labels)
        return self.shapelet_explainer
    
    def plot_explanation(self, 
                         rules = True, 
                         heatmap = False, 
                         latent_space = True,
                         VAE_2d = False,
                         shap_explanation = True,
                         multi_shap_explanation = False,
                         shapelet_explanation = True,
                         figsize = (20,3),
                         **params
                         ):
        # params.keys = [nsamples, background, pen, peltmodel, 
        #               jump, graph_out_file, shapelet_mapper, VAE_2d_grid_size,
        #               l, r, optimizer, weight_regularizer, max_iter]
        
        # plot instance to explain
        plt.figure(figsize = figsize)
        if not self.labels:
            plt.title(label = "Instance to Explain, class: " + str(self.instance_to_explain_blackbox_class))
        else:
            plt.title(label = "Instance to Explain, class: " + self.labels[self.instance_to_explain_blackbox_class] + " (" + str(self.instance_to_explain_blackbox_class)+")")
        plt.plot(self.instance_to_explain)
        plt.show()
        
        # plot rules and crules
        if rules: self.plot_rules_dataframes(figsize = figsize)
        
        # plot heatmaps
        if heatmap: self.plot_rules_heatmaps(figsize = figsize)
        
        # plot shap explanation on rules and crules centroids
        #self.shap_output_data = []
        if shap_explanation:
            medoid_df = []
            for rule in self.rules_dataframes.keys():
                medoid_df.append(self.rules_dataframes[rule]["df"][self.rules_dataframes[rule]["medoid_idx"]])
            medoid_df = np.array(medoid_df)
            rules_shap_values = []
            rules_segmentations = []
            for i, medoid_ts in enumerate(medoid_df):
                print(list(self.rules_dataframes.keys())[i])
                shap_values, segmentation = self.shap_ts(ts = medoid_ts, 
                                                         classifier = self.blackbox, 
                                                         input_dim = self.blackbox_input_dimensions, 
                                                         figsize = figsize, 
                                                         nsamples = params.get("nsamples", 1000),
                                                         background = params.get("background", "linear"),
                                                         pen = params.get("pen", 1),
                                                         model = params.get("peltmodel", "rbf"),
                                                         jump = params.get("jump", 5)
                                                         )
                if params.get("shap_by_class", False):
                    rules_shap_values.append(shap_values)
                    rules_segmentations.append(segmentation)
                self.plot_shap(ts = medoid_ts, 
                               shap_values = shap_values, 
                               segmentation = segmentation, 
                               figsize = figsize)
            if params.get("shap_by_class", False):
                print("SHAP by class")
                self.plot_shap_by_class(medoid_df,
                                        rules_shap_values,
                                        rules_segmentations,
                                        figsize = figsize
                                        )
        
        if multi_shap_explanation:
            for rule in self.rules_dataframes.keys():
                print(rule)
                self.multi_shap(self.rules_dataframes[rule]["df"],
                                medoid_idx = self.rules_dataframes[rule]["medoid_idx"],
                                n = params.get("multishap_n", -1), 
                                figsize = figsize, 
                                nsamples = params.get("nsamples", 1000),
                                background = params.get("background", "linear"),
                                pen = params.get("pen", 1),
                                model = params.get("peltmodel", "rbf"),
                                jump = params.get("jump", 5),
                                )
                
    
        
        # plot shapelet explanation on rules and crules centroids
        if shapelet_explanation:
            if self.shapelet_explainer is None:
                self.build_shapelet_explainer(l = params.get("l",0.1),
                                              r = params.get("r",2), 
                                              weight_regularizer = params.get("weight_regularizer", .01),
                                              optimizer = params.get("optimizer", "sgd"),
                                              random_state = params.get("random_state", None),
                                              max_iter = params.get("max_iter", 100))
            medoid_df = []
            for rule in self.rules_dataframes.keys():
                medoid_df.append(self.rules_dataframes[rule]["df"][self.rules_dataframes[rule]["medoid_idx"]])
            medoid_df = np.array(medoid_df)
            for i, medoid_ts in enumerate(medoid_df):
                print(list(self.rules_dataframes.keys())[i])
                self.shapelet_explainer.plot_series_shapelet_explanation(medoid_ts,
                                                                         self.blackbox_predict(medoid_ts.reshape(1,-1,1)),
                                                                         figsize = figsize
                                                                         )
                self.shapelet_explainer.fidelity = accuracy_score(self.Zy_latent_instance_neighborhood_labels,
                                        self.shapelet_explainer.predict(self.Z_latent_instance_neighborhood_decoded))
            
        # plot a visualization of the latent space
        if latent_space:
            blackbox_dataset = self.X_explanation.copy()
            blackbox_labels = self.blackbox_predict(blackbox_dataset)
            dataset_latent = self.X_explanation_latent
            
            self.visualize_latent_space(dataset_latent = dataset_latent, 
                                        dataset_labels = blackbox_labels, 
                                        neighborhood_plot = True,
                                        rules_plot = True,
                                        pca = True, 
                                        )
        # plot the VAE normal generation (meaningful only with VAE and 2d latent space)
        if VAE_2d:
            self.VAE_normal_2dgeneration(n = params.get("VAE_2d_grid_size", 9), figsize = (20,10))
    
    def plot_2dlatent_space(self,
                          dataset_latent, 
                          dataset_labels, 
                          instance_to_explain_latent,
                          latent_neighborhood = None,
                          latent_neighborhood_labels = None,
                          rules_dataframes_latent = None,
                          figsize = (20, 6), 
                          #shap_plot = False, 
                          neighborhood_plot = True,
                          rules_plot = False):
        
        fig, ax = plt.subplots(figsize=figsize)
        fig.suptitle("Latent Space")
        
        # plots dataset latent points
        scatter = ax.scatter(dataset_latent[:, 0], dataset_latent[:, 1], c = dataset_labels, cmap = "Set1")
        ax.legend(*scatter.legend_elements(), loc="lower left", title="Classes")
        
        # plots generated neighborhood points
        if neighborhood_plot:
            ax.scatter(latent_neighborhood[:,0], 
                       latent_neighborhood[:,1], 
                       c = latent_neighborhood_labels, 
                       alpha = 0.5, marker = ".", cmap = "Set1")
        
        # marks with a black circle the points covered by a rule or a counterfactual
        if rules_plot:
            for i, rule in enumerate(rules_dataframes_latent.keys()):
                ax.scatter(rules_dataframes_latent[rule]["df"][:,0], 
                           rules_dataframes_latent[rule]["df"][:,1], 
                           alpha = 1, 
                           marker = "o",
                           label = rule,
                           facecolors='none', edgecolors="black"
                           )
        
        # plots shap generated points (it's a mess because the autoencoder is not trained to encode them)
        """
        if shap_plot:
            for rule_index, data in enumerate(shap_output_data):
                shap_y = np.argmax(blackbox.predict(data.reshape(data.shape[0], data.shape[1], 1)), axis = 1)
                shap_lat = encoder.predict(data.reshape(data.shape[0],data.shape[1], 1))
                plt.scatter(shap_lat[:,0], shap_lat[:,1], c = shap_y, alpha = 0.2, marker = "*", cmap = "Set1")
        """
        """instance_to_explain_latent = dataset_latent[self.index_to_explain].ravel()"""
        
        # marks the instance to explain with an X
        ax.scatter(instance_to_explain_latent[0], 
                   instance_to_explain_latent[1], 
                   c = "black", marker = "x", s = 300)
        plt.show()
        
    def visualize_latent_space(self, 
                               dataset_latent, 
                               dataset_labels, 
                               neighborhood_plot = True,
                               rules_plot = False,
                               pca = False
                               ):
        
        # if the latend space is 2d we can visualize it directly
        if self.Z_latent_instance_neighborhood.shape[1] == 2:
            self.plot_2dlatent_space(dataset_latent, dataset_labels, 
                                     latent_neighborhood = self.Z_latent_instance_neighborhood, 
                                     latent_neighborhood_labels = self.Zy_latent_instance_neighborhood_labels,
                                     instance_to_explain_latent = self.instance_to_explain_latent,
                                     rules_dataframes_latent = self.rules_dataframes_latent,
                                     figsize = (20, 8), #shap_plot = False, 
                                     neighborhood_plot = neighborhood_plot,
                                     rules_plot = False)
            if rules_plot:
                self.plot_2dlatent_space(dataset_latent, dataset_labels, 
                                         latent_neighborhood = self.Z_latent_instance_neighborhood, 
                                         latent_neighborhood_labels = self.Zy_latent_instance_neighborhood_labels,
                                         instance_to_explain_latent = self.instance_to_explain_latent,
                                         rules_dataframes_latent = self.rules_dataframes_latent,
                                         figsize = (20, 8), #shap_plot = False, 
                                         neighborhood_plot = neighborhood_plot,
                                         rules_plot = True)
                
        # if the latent space is multidimensional we must use pca first
        else:
            if pca:
                pca_2d = PCA(n_components=2)
                pca_2d.fit(dataset_latent)
                dataset_latent_2dconversion = pca_2d.transform(dataset_latent)
                instance_to_explain_latent = pca_2d.transform(self.instance_to_explain_latent.reshape(1,-1)).ravel()
                neighborhood_latent_2dconversion = None
                rules_dataframes_latent_2dconversion = None
                if neighborhood_plot:
                    neighborhood_latent_2dconversion = pca_2d.transform(self.Z_latent_instance_neighborhood)
                if rules_plot:
                    rules_dataframes_latent_2dconversion = dict()
                    for rule in self.rules_dataframes_latent.keys():
                        rules_dataframes_latent_2dconversion[rule] = {"df":pca_2d.transform(self.rules_dataframes_latent[rule]["df"])}
                self.plot_2dlatent_space(dataset_latent_2dconversion, dataset_labels, 
                                         latent_neighborhood = neighborhood_latent_2dconversion, 
                                         latent_neighborhood_labels = self.Zy_latent_instance_neighborhood_labels,
                                         instance_to_explain_latent = instance_to_explain_latent,
                                         rules_dataframes_latent = rules_dataframes_latent_2dconversion,
                                         figsize = (20, 8), #shap_plot = False, 
                                         neighborhood_plot = neighborhood_plot,
                                         rules_plot = False)
                if rules_plot:
                    self.plot_2dlatent_space(dataset_latent_2dconversion, dataset_labels, 
                                             latent_neighborhood = neighborhood_latent_2dconversion, 
                                             latent_neighborhood_labels = self.Zy_latent_instance_neighborhood_labels,
                                             instance_to_explain_latent = instance_to_explain_latent,
                                             rules_dataframes_latent = rules_dataframes_latent_2dconversion,
                                             figsize = (20, 8), #shap_plot = False, 
                                             neighborhood_plot = neighborhood_plot,
                                             rules_plot = True)
            
                        
    def VAE_normal_2dgeneration(self, n = 9, figsize = (20,10)):
        grid_x = norm.ppf(np.linspace(0.05, 0.95, n)) 
        grid_y = norm.ppf(np.linspace(0.05, 0.95, n)) 
        fig, axs = plt.subplots(n, n, figsize=figsize)
        fig.suptitle("VAE generation")
        fig.patch.set_visible(False)
        colors = ["r", "g", "blue", "c", "m", "k", "orange", "olive", "pink"]
        for i, yi in enumerate(grid_x):
            for j, xi in enumerate(grid_y):
                z_sample = np.array([[xi, yi]])
                x_decoded = self.decoder.predict(z_sample).ravel()
                x_label = self.blackbox_predict(x_decoded.reshape(1,-1,1))[0]
                axs[i,j].plot(x_decoded, color = colors[x_label%len(colors)], 
                   label = "class: " + str(x_label) if not self.labels else "class: " + str(self.labels[x_label]) + " ({})".format(str(x_label)))
                axs[i,j].set_yticklabels([])
                axs[i,j].set_xticklabels([])
                axs[i,j].axis('off')
                
        d = dict()
        for a in fig.get_axes():
            if a.get_legend_handles_labels()[1][0] not in d:
                d[a.get_legend_handles_labels()[1][0]] = a.get_legend_handles_labels()[0][0]
                
        labels, handles = zip(*sorted(zip(d.keys(), d.values()), key=lambda t: t[0]))
        plt.legend(handles, labels)
    
    
    """
    def build_dataset_shapelet_mapper(self, dataset_transformed):
        minima = dataset_transformed.min()
        maxima = dataset_transformed.max()
    
        norm = matplotlib.colors.LogNorm(vmin=minima, vmax=maxima)
        mapper = cm.ScalarMappable(norm=norm, cmap=cm.Reds_r)
        return mapper
    """
                
            
if __name__ == '__main__':
    from pyts.datasets import make_cylinder_bell_funnel
    from sklearn.model_selection import train_test_split
    from autoencoders import Autoencoder
    from joblib import load, dump
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
    
    
    knn = load("./blackbox_checkpoints/cbf_blackbox_knn_20191106_145654.joblib")
    
    blackbox = build_resnet(n_timesteps, n_outputs)
    blackbox.load_weights("./blackbox_checkpoints/cbf_blackbox_resnet_20191106_145242_best_weights_+1.00_.hdf5")
    resnet = blackbox
    
    # problems with LambdaLayer pickle
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
    #autoencoder.load_weights("./autoencoder_checkpoints/cbf_autoencoder_20191106_144909_best_weights_+136.8745_.hdf5")
    
    
    
    index_to_explain = 0
    blackbox = resnet
    encoder = autoencoder.layers[1]
    decoder = autoencoder.layers[2]
    blackbox_input_dimensions = 3
    
    print("\nEXPLAINER")
    agnostic = AgnosticLocalExplainer(blackbox, 
                                  encoder, 
                                  decoder, 
                                  autoencoder,  
                                  X_explanation = X_exp_test, 
                                  y_explanation = y_exp_test, 
                                  index_to_explain = index_to_explain,
                                  blackbox_input_dimensions = blackbox_input_dimensions,
                                  labels = ["cylinder", "bell", "funnel"]
                                 )
    agnostic.check_autoencoder_blackbox_consistency()
    print("\nNeighborhood Generation")
    agnostic.LOREM_neighborhood_generation(
                          neigh_type = 'geneticp', 
                          categorical_use_prob = True,
                          continuous_fun_estimation = False, 
                          size = 100,
                          ocr = 0.1, 
                          multi_label=False,
                          one_vs_rest=False,
                          verbose = True,
                          ngen = 2)
    print("\nExtracting Rules")
    agnostic.LOREM_tree_rules_extraction()
    agnostic.build_rules_dataframes()
    
    
    """
    agnostic.print_rules_n()
    agnostic.LOREM_rules_random_augmentation(rules_to_augment = ["rule"], size = 100)
    agnostic.print_rules_n()
    """
    
    params = {"background": "linear_consecutive", 
              "nsamples":100, 
              "shap_by_class" : False,
              #"optimizer": keras.optimizers.Adam(),#keras.optimizers.Adagrad(lr=.1), 
              "multishap_n":10}
    
    agnostic.plot_explanation( 
                         rules = True, 
                         heatmap = False, 
                         shap_explanation = False, 
                         shapelet_explanation = True,
                         latent_space = False,
                         multi_shap_explanation = False,
                         figsize = (20,3),
                         VAE_2d = False,
                         **params
                         )
    
    