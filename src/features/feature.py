"""
This module gets all of the features and randomly select the sample with the size of dimension

parameters:
* dimension : (int) an integer number specifies total number of features that will be generate in the data set
* file_path : (string) file path to supply to data_utilities

returns:
type : numpy array
* array of string with the size of given dimension
"""

import numpy as np
import random
from data import data_utilities

def get_all_features(dimension, file_path):
    all_features = data_utilities.get_features(data_utilities.read_file(file_path))

    if all_features.shape[0] < dimension:
        raise Exception('Program terminated, number of dimension is higher than total data.')
    
    return np.array(random.sample(all_features.tolist(), dimension))