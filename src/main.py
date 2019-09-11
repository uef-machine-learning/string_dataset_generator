"""
Main flow of artificial set data generator program
This module should be treated as a client of the library
"""

import artificial_set_data_generator as dg
from features import feature


# Parameters setting
DATA_SIZE = 100
SIZE_OF_SET = (4, 20) # [NOT IMPLEMENTED] tuple of large and small data set
NUMBER_OF_CLUSTER = 5
DIMENSION = 200
DISTANCE_THRESHOLD = 0.2
MINIMUM_FEATURE_PER_ENTRY = 4
FILE_PATH = '../data/50000.txt'
ALL_FEATURES = feature.get_all_features(DIMENSION, FILE_PATH)

# Calling the library
dg.generate(
    DATA_SIZE, 
    SIZE_OF_SET, 
    NUMBER_OF_CLUSTER, 
    DIMENSION, 
    DISTANCE_THRESHOLD, 
    MINIMUM_FEATURE_PER_ENTRY,
    ALL_FEATURES)

