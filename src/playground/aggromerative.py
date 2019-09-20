import numpy as np
from distances import distance_calculator
from itertools import combinations

def cluster(data, number_of_cluster, pw_dist):
    if number_of_cluster == 0:
        number_of_cluster = 2
    # assign each data into its own cluster
    labels = np.arange(len(data))
    total_within_cluster_variance = {}

    # find closest distance and merge them
    np.fill_diagonal(pw_dist, np.inf)
    cur_number_of_cluster = len(np.unique(labels))
    while(cur_number_of_cluster > 2):
        
        # merge pair
        indices = divmod(pw_dist.argmin(), pw_dist.shape[1])
        #print(indices)
        #print('label: 0',labels[indices[0]])
        #print('label: 1', labels[indices[1]])
        
        effected_labels = np.where(labels == labels[indices[1]])[0]
        labels[effected_labels] = labels[indices[0]]
        #labels[indices[1]] = labels[indices[0]]

        #print('label: 0', labels[indices[0]])
        #print('label: 1', labels[indices[1]])

        # update distance to inf
        # TODO : this can be improved
        # bug here 961
        effected_labels = np.where(labels == labels[indices[0]])[0]
        iterator = combinations(effected_labels, 2)

        for i, j in iterator:   
            pw_dist[i, j] = np.inf
            pw_dist[j, i] = np.inf

        # get current number of cluster
        #print(labels)
        cur_number_of_cluster = len(set(labels))
        print(cur_number_of_cluster)
        #print('=================')


