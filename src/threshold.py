import numpy as np
def mean_std(predictions,**params):
    mean = np.mean(predictions, axis=0)
    std = np.std(predictions, axis=0)
    threshold = mean + params['t'] * std
    return(threshold)
