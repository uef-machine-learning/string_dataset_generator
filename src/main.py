"""
Main flow of artificial set data generator program
This module should be treated as a client of the library
"""

import artificial_set_data_generator as dg
from features import feature
from data import data_utilities
from imbalance import size
import numpy as np

# Parameters setting
DATA_SIZE = 1000
NUMBER_OF_CLUSTER = 6
SIZE_OF_CLUSTERS = [] #size.random_cluster_sizes(DATA_SIZE, NUMBER_OF_CLUSTER)
DIMENSION = 200
DISTANCE_THRESHOLD = 0.85
SIZE_OF_SET = (4,20)
ALL_FEATURE_FILE_PATH = '../data/50000.txt'
GT_REPRESENTATIVE_FILE_PATH = '../data/1000N_6K_gt_representative.txt'


# Read ground truth representative
if GT_REPRESENTATIVE_FILE_PATH != '': 
    gt_representative = data_utilities.read_file(GT_REPRESENTATIVE_FILE_PATH)
else:
    gt_representative = []

all_features = feature.get_all_features(DIMENSION, ALL_FEATURE_FILE_PATH, gt_representative)

# Calling the library
data, ground_truth_labels, representatives, overlap_percentage = dg.generate(
    DATA_SIZE, 
    SIZE_OF_CLUSTERS, 
    NUMBER_OF_CLUSTER, 
    DIMENSION, 
    DISTANCE_THRESHOLD, 
    SIZE_OF_SET,
    all_features,
    gt_representative)

print('--- overlap percentage is:', overlap_percentage)

OUT_DATA_FILE_NAME = '../out/gen_data{}N_{}K_{}overlap.txt'.format(DATA_SIZE, NUMBER_OF_CLUSTER, overlap_percentage)
OUT_GT_REPRESENTATIVE_FILE_NAME = '../out/gt_representative{}N_{}K_{}overlap.txt'.format(DATA_SIZE, NUMBER_OF_CLUSTER, overlap_percentage)
OUT_GT_LABELS_FILE_NAME = '../out/gt_labels{}N_{}K_{}overlap.txt'.format(DATA_SIZE, NUMBER_OF_CLUSTER, overlap_percentage)

data_utilities.write_file(data, OUT_DATA_FILE_NAME)
data_utilities.write_file(representatives, OUT_GT_REPRESENTATIVE_FILE_NAME)
np.savetxt(OUT_GT_LABELS_FILE_NAME, ground_truth_labels.T, fmt='%d') 
