# the code is largely adopted from https://github.com/seinan9/LSCDiscovery/blob/main/measures/apd.py
import logging
import random
import sys
sys.path.append('./modules')
import time

from docopt import docopt
import numpy as np
from scipy.spatial.distance import cosine as cosine_distance

from utils_ import Space
def apd(path_matrix1,path_matrix2):

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    logging.info(__file__.upper())
    start_time = time.time()

    # Load matrices
    try:
        space1 = Space(path_matrix1, format='npz')
    except ValueError:
        space1 = Space(path_matrix1, format='w2v')
    try:
        space2 = Space(path_matrix2, format='npz')
    except ValueError:
        space2 = Space(path_matrix2, format='w2v')

    vectors1 = space1.matrix.toarray()
    vectors2 = space2.matrix.toarray()
    '''
    # Get number of rows/vectors
    samples_corpus1 = []
    samples_corpus2 = []

    # Set the sample size and get the samples
    if len(vectors1) > len(vectors2):
        max_ = len(vectors1)
        min_ = len(vectors2)
        samples_corpus2 = vectors2
        randoms = random.sample(range(max_), min_)
        for i in randoms:
            samples_corpus1.append(vectors1[i])
    else:
        max_ = len(vectors2)
        min_ = len(vectors1)
        samples_corpus1 = vectors1
        randoms = random.sample(range(max_), min_)
        for i in randoms:
            samples_corpus2.append(vectors2[i])
    '''
    # Compute the average pairwise cosine distance
    apds = []
    for i in range(0, len(vectors1)):
        for j in range(0, len(vectors1)):
            apd=cosine_distance(vectors1[i], vectors2[j])
            #apd=cosine_distance(samples_corpus1[i], samples_corpus2[j])
            apds.append(apd)
    apd = np.mean(apds, axis=0)

    # Print output
    print(apd)
    return(apd)

    logging.info("--- %s seconds ---" % (time.time() - start_time))
    print("")
