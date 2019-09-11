import numpy as np
from itertools import chain

def read_file():
    data_list = []

    with open('../data/50000.txt', 'r') as f:
        for line in f.readlines():
            line = line.rstrip()
            line = line.split(' ')
            data_list.append(line)

    return np.array(data_list)

def get_features(np_data):
    np_data_1d = list(chain.from_iterable(np_data))
    unique_set = set(np_data_1d)
    unique_set.discard('')
    features = np.array(list(unique_set))
    return features