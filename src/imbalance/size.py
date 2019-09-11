"""
This module randomly create size to clusters that the sum of sizes equal to total data size

paremeters:
n : (int) number of total sum (data size)
num_terms : number of cluster
"""

import random as r
import numpy as np

def random_cluster_sizes(n, num_terms = None):
    num_terms = (num_terms or r.randint(2, n)) - 1
    a = r.sample(range(1, n), num_terms) + [0, n]
    list.sort(a)
    x = np.array([a[i+1] - a[i] for i in range(len(a) - 1)])
    print(x)
    return x
