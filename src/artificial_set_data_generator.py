"""
This module contain the implementation of artificial data generator for set data

parameters:
* data_size : (int) an integer number specifies number of total number of data that will be generated
* size_of_set : [NOT IMPLEMENTED](tuple) specifies min and max number of features per each data
* number_of_cluster : (int) an integer number specifies number of cluster to create
* dimension : (int) an integer number specifies total number of features that will be generate in the data set
* distance_threshold : (float) a number specifies the maximum distance away from the cluster representative according to Jaccard's method
* minimum_feature_per_entry : (int) an interger specifies the minimum feature that each data has to contain
* all_features : (string[]) an array of string containing all possible features of the dataset

returns:
type : tuple
* artificially generated data set as a list of numpy array containing strings
* ground truth of the data set
"""

import math
import numpy as np
import random
from itertools import chain, combinations_with_replacement
from entries import entry



def _remove_assigned_features(cp_all_features, representative):
    return np.array([x for x in cp_all_features if x not in representative])

def _calculate_pairwise_distance(X):
    # TODO : this function is very slow need improvement
    data_size = X.shape[0]
    precomputed = np.zeros((data_size, data_size))
    iterator = combinations_with_replacement(range(X.shape[0]), 2)

    for i, j in iterator:
        precomputed[i, j] = _jaccard_seq(X[i], X[j])     

    # Make symmetric and return
    return precomputed + precomputed.T - np.diag(np.diag(precomputed))

def _find_closest_member_from_other_clusters(own_cluster_id, pw_dist, ground_truths, data_id):
    all_member_from_other_cluster_ids = np.where(ground_truths != own_cluster_id)[0]

    pw_excluded_own_cluster = np.full((1, pw_dist.shape[1]), np.Inf)
    pw_excluded_own_cluster[:, all_member_from_other_cluster_ids] = pw_dist[data_id, all_member_from_other_cluster_ids]

    min_index = np.argmin(pw_excluded_own_cluster)
    return min_index

def _calculate_overlap(data, ground_truths, representatives, pw_dist):
    overlap_count = 0
    for i in range(len(data)):
        closest_member_from_other_cluster_id = _find_closest_member_from_other_clusters(ground_truths[i], pw_dist, ground_truths, i)
        closest_member_from_other_cluster = data[closest_member_from_other_cluster_id]
        if _jaccard_seq(data[i], closest_member_from_other_cluster) < _jaccard_seq(data[i], representatives[ground_truths[i]]):
            print('there is an overlap at data', i, 'with cluster', ground_truths[closest_member_from_other_cluster_id])
            overlap_count = overlap_count + 1
    print('overlap count is:', overlap_count)
    print('overlap percentage is:', (overlap_count * 100) / len(data))

def generate(
    data_size, 
    size_of_set, 
    number_of_cluster, 
    dimension, 
    distance_threshold, 
    minimum_feature_per_entry,
    all_features):

    representatives = entry.create_cluster_representatives(
        number_of_cluster, 
        minimum_feature_per_entry, 
        all_features)
    print('=== done representative calculation ===')

    data, ground_truths = _generate_cluster_members(
        data_size, 
        size_of_set, 
        representatives, 
        minimum_feature_per_entry, 
        all_features,
        distance_threshold)
    print('=== done generating data and ground truths ===')

    pw_dist = _calculate_pairwise_distance(np.array(data))
    print('=== done pairwise distance calculation ===')

    _calculate_overlap(data, ground_truths, representatives, pw_dist)
    print('=== done overlap calculation ===')