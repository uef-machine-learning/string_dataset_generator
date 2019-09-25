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
    return np.array([a[i+1] - a[i] for i in range(len(a) - 1)])

def build_specific_sizes(n, number_of_big_and_small_cluster, ratio):
    number_big = number_of_big_and_small_cluster[0]
    number_small = number_of_big_and_small_cluster[1]

    small_member = n / ((number_big * ratio) + number_small)
    big_member = ratio * small_member

    small_list = [small_member] * number_small
    big_list = [big_member] * number_big

    total_list = small_list + big_list
    r.shuffle(total_list)
    
    return np.array(total_list)
