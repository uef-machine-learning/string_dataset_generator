import numpy as np
import matplotlib.pyplot as plt

NUM_EXAMPLES = 100000
SEQ_LENGTH = 10

sequences = np.zeros((NUM_EXAMPLES, SEQ_LENGTH), dtype=np.int8)
# How many number of ones in each sequence
number_of_1s = np.random.randint(0, SEQ_LENGTH+1, size=NUM_EXAMPLES)

indices = np.arange(SEQ_LENGTH)
for idx, num_ones in enumerate(number_of_1s.tolist()):
    # Set "num_ones" elements to 1 using "choice" without replace.
    sequences[idx][np.random.choice(indices, num_ones, replace=False)] = 1

print(sequences)
plt.hist(np.sum(sequences==1, axis=1), bins=np.arange(SEQ_LENGTH+2)-0.5, histtype='step')
plt.show()