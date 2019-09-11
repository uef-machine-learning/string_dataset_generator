"""
This module contain the implementation of artificial data generator for set data

parameters:
* data_size : (int) an integer number specifies number of total number of data that will be generated
* size_of_set : (tuple) specifies min and max number of features per each data
* number_of_cluster : (int) an integer number specifies number of cluster to create
* dimension : (int) an integer number specifies total number of features that will be generate in the data set
* overlap : (enum) specifies cluster overlap percentage
* imbalance : (tuple) specifies number of big and small cluster

returns:
* artificially generated data set as a list of numpy array containing strings
"""

import data_utilities
import random
import numpy as np
from itertools import chain, combinations_with_replacement
import math

def _build_artificial_features(dimension):
    all_features = data_utilities.get_features(data_utilities.read_file())

    if all_features.shape[0] < dimension:
        raise Exception('Program terminated, number of dimension is higher than total data.')
    
    return np.array(random.sample(all_features.tolist(), dimension))

def _create_cluster_member_random(all_features, min_feature, max_feature):
    number_of_member_features = random.randrange(min_feature, max_feature + 1)
    return random.sample(all_features.tolist(), number_of_member_features)

def _create_cluster_member_with_one_different(representative, all_features):
    cp_representative = np.copy(representative)
    feature_id_to_change = random.randint(1, len(representative) - 1)
    new_feature = random.sample(all_features.tolist(), 1)[0]

    cp_representative[feature_id_to_change] = new_feature

    return cp_representative

def _create_cluster_member_random_but_same_feature_length(representative, all_features):
    cp_representative = np.copy(representative)

    # How many number of ones in each sequence
    selected_features = np.random.randint(0, 2, size=len(representative))
    feature_ids_to_change = np.where(selected_features == 1)[0]

    if feature_ids_to_change.shape[0] == 0:
        return _create_cluster_member_with_one_different(representative, all_features)
    else:
        new_features = random.sample(all_features.tolist(), feature_ids_to_change.shape[0])[0]
        cp_representative[feature_ids_to_change] = new_features

        return cp_representative

def _create_cluster_member_random_different_feature_length(
    representative, 
    all_features, 
    min_feature, 
    max_feature, 
    distance_threshold):
    jaccard_similarity_threshold = 1 - distance_threshold
    number_of_member_features = random.randrange(min_feature, (max_feature - len(representative)) + 1)
    min_union = math.floor((jaccard_similarity_threshold * (len(representative) + number_of_member_features)) / (1 + jaccard_similarity_threshold))

    while(min_union >= number_of_member_features or min_union >= len(representative)):
        number_of_member_features = random.randrange(min_feature, max_feature + 1)
        min_union = math.floor((jaccard_similarity_threshold * (len(representative) + number_of_member_features)) / (1 + jaccard_similarity_threshold))

    new_member = np.full(number_of_member_features, '   ')
    number_of_union = np.random.randint(min_union, min(number_of_member_features, len(representative)))
    total_available_features = [x for x in all_features if x not in representative]
    
    if number_of_member_features > len(representative):
        new_member[0:len(representative)] = representative
        id_to_change = random.sample(range(len(representative)), len(representative) - number_of_union)
        empty_space = len(np.where(new_member == '   ')[0])
        new_features_pool = random.sample(total_available_features, len(id_to_change) + empty_space)
        new_member[len(representative):] = new_features_pool[0:(number_of_member_features - len(representative))]
        new_member[id_to_change] = new_features_pool[(number_of_member_features - len(representative)):]
        # print('representative:', representative)
        # print('current new member:', new_member)
        # print('number of total new feature pool:', len(id_to_change) + empty_space)
    elif number_of_member_features < len(representative):
        selected_id = random.sample(range(len(representative)), number_of_member_features)
        new_member = representative[selected_id]
        new_features = random.sample(total_available_features, len(new_member) - number_of_union)
        id_to_change = random.sample(range(len(new_member)), len(new_member) - number_of_union)
        new_member[id_to_change] = new_features
        # print('representative:', representative)
        # print('current new member:', new_member)
    else:
        new_member = np.copy(representative)
        id_to_change = random.sample(range(number_of_member_features), number_of_member_features - number_of_union)
        new_features = random.sample(total_available_features, len(id_to_change))
        new_member[id_to_change] = new_features
        # print('representative:', representative)
        # print('current new member:', new_member)
    return new_member

def _remove_assigned_features(cp_all_features, representative):
    return np.array([x for x in cp_all_features if x not in representative])
    
def _create_cluster_unique_representatives(number_of_cluster, min_member, all_features):
    cp_all_features = np.copy(all_features)
    max_feature_per_cluster = int(all_features.shape[0] / number_of_cluster)
    
    all_representative = []
    for i in range(number_of_cluster):
        representative = np.array(_create_cluster_member_random(cp_all_features, min_member, max_feature_per_cluster))
        all_representative.append(representative)
        cp_all_features = _remove_assigned_features(cp_all_features, representative)

    return all_representative

def _create_cluster_representatives(number_of_cluster, min_member, all_features):
    max_feature_per_cluster = int(all_features.shape[0] / number_of_cluster)
    
    all_representative = []
    for i in range(number_of_cluster):
        representative = np.array(_create_cluster_member_random(all_features, min_member, max_feature_per_cluster))
        all_representative.append(representative)

    return all_representative


def _get_number_of_intersec_data(x, y):
    return len(set(x) & set(y))

def _get_number_of_union_data(x, y):
    return len(set(x) | set(y))

def _jaccard_seq(x, y):
    return 1 - (_get_number_of_intersec_data(x, y) / _get_number_of_union_data(x, y))

def _find_number_of_member_per_cluster(data_size, representatives):
    n_centers = len(representatives)
    number_of_data_per_cluster = [int((data_size) // n_centers)] * n_centers

    for i in range(data_size % n_centers):
        number_of_data_per_cluster[i] += 1

    return number_of_data_per_cluster


def _generate_cluster_members(
    data_size, 
    size_of_set, 
    representatives, 
    min_member,
    all_features):
    jaccard_threshold = 0.2
    number_of_data_per_cluster = _find_number_of_member_per_cluster(data_size, representatives)
    data = []
    clusters = []

    for i in range(len(representatives)):
        cluster_len = 0

        data.append(representatives[i])
        clusters.append(i)
        cluster_len = 1

        while cluster_len != number_of_data_per_cluster[i]:
            # member = _create_cluster_member_random(all_features, 4, len(all_features))
            #member = _create_cluster_member_random_but_same_feature_length(representatives[i], all_features)
            member = _create_cluster_member_random_different_feature_length(representatives[i], all_features, min_member, len(all_features), jaccard_threshold)

            if _jaccard_seq(representatives[i], member) < jaccard_threshold:
                data.append(member)
                clusters.append(i)
                cluster_len = cluster_len + 1

    if len(np.array(clusters)) != len(data):
        raise Exception('Program terminated, lengths of data and ground truths are not equal')

    return (data, np.array(clusters))

def _calculate_pairwise_distance(X):
    # this function is very slow
    # need improvement
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
    overlap_list = np.zeros(len(data))
    for i in range(len(data)):
        closest_member_from_other_cluster_id = _find_closest_member_from_other_clusters(ground_truths[i], pw_dist, ground_truths, i)
        closest_member_from_other_cluster = data[closest_member_from_other_cluster_id]
        if _jaccard_seq(data[i], closest_member_from_other_cluster) < _jaccard_seq(data[i], representatives[ground_truths[i]]):
            print('there is an overlap at data', i, 'with cluster', ground_truths[closest_member_from_other_cluster_id])

def generate(
    data_size, 
    size_of_set, 
    number_of_cluster, 
    dimension, 
    overlap, 
    imbalance, 
    min_member):
    all_features = _build_artificial_features(dimension)

    representatives = _create_cluster_representatives(number_of_cluster, min_member, all_features)
    print('=== done representative calculation ===')

    data, ground_truths = _generate_cluster_members(data_size, size_of_set, representatives, min_member, all_features)
    print('=== done generating data and ground truths ===')

    pw_dist = _calculate_pairwise_distance(np.array(data))
    print('=== done pairwise distance calculation ===')

    _calculate_overlap(data, ground_truths, representatives, pw_dist)
    print('=== done overlap calculation ===')