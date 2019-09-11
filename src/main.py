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
DATA_SIZE = 100
NUMBER_OF_CLUSTER = 5
SIZE_OF_SET = size.random_cluster_sizes(DATA_SIZE, NUMBER_OF_CLUSTER)# [NOT IMPLEMENTED] tuple of large and small data set
DIMENSION = 200
DISTANCE_THRESHOLD = 0.8
MINIMUM_FEATURE_PER_ENTRY = 4
FILE_PATH = '../data/50000.txt'
ALL_FEATURES = feature.get_all_features(DIMENSION, FILE_PATH)

# Calling the library
data, ground_truths, representatives = dg.generate(
    DATA_SIZE, 
    SIZE_OF_SET, 
    NUMBER_OF_CLUSTER, 
    DIMENSION, 
    DISTANCE_THRESHOLD, 
    MINIMUM_FEATURE_PER_ENTRY,
    ALL_FEATURES)

data_utilities.write_file(data, '../out/gen_data.txt')
data_utilities.write_file(representatives, '../out/gen_representative.txt')
np.savetxt('../out/gen_ground_truths.txt', ground_truths.T, fmt='%d') 
