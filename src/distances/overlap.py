"""
This module calculate overlap of the generate data set

'overlap' is defined as if a point in cluster A has the distance to its medoids further that the distance of its closest member from other clusters

parameters:
* data : artificially generated data set as a list of numpy array containing strings
* ground_truth_labels : (numpy array) ground truths labels  of the data set
* representatives : list of numpy array (cluster representative / medoids)
"""

from distances import jaccard, pairwise
import numpy as np

def _find_closest_member_from_other_clusters(own_cluster_id, pw_dist, ground_truth_labels, data_id):
    all_member_from_other_cluster_ids = np.where(ground_truth_labels != own_cluster_id)[0]

    pw_excluded_own_cluster = np.full((1, pw_dist.shape[1]), np.Inf)
    pw_excluded_own_cluster[:, all_member_from_other_cluster_ids] = pw_dist[data_id, all_member_from_other_cluster_ids]

    min_index = np.argmin(pw_excluded_own_cluster)
    return min_index

def calculate_overlap(data, ground_truth_labels, representatives):
    pw_dist = pairwise.calculate_pairwise_distance(np.array(data))
    print('=== done pairwise distance calculation ===')

    overlap_count = 0
    for i in range(len(data)):
        closest_member_from_other_cluster_id = _find_closest_member_from_other_clusters(ground_truth_labels[i], pw_dist, ground_truth_labels, i)
        closest_member_from_other_cluster = data[closest_member_from_other_cluster_id]
        if jaccard.jaccard_seq(data[i], closest_member_from_other_cluster) < jaccard.jaccard_seq(data[i], representatives[ground_truth_labels[i]]):
            print('there is an overlap at data', i, 'with cluster', ground_truth_labels[closest_member_from_other_cluster_id])
            print('id of the overlap partner is ', closest_member_from_other_cluster_id)
            overlap_count = overlap_count + 1

    print('overlap percentage is:', (overlap_count * 100) / len(data))