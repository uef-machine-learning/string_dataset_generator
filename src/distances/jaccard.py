"""
This module calculate Jaccard's distance score

parameters:
x : (numpy array) first string sequence
y : (numpy array) second string sequence

return:
(float) : Jaccard's distance socre (1 - Jaccard's similarlity)
"""
def _get_number_of_intersec_data(x, y):
    return len(set(x) & set(y))

def _get_number_of_union_data(x, y):
    return len(set(x) | set(y))

def jaccard_seq(x, y):
    return 1 - (_get_number_of_intersec_data(x, y) / _get_number_of_union_data(x, y))