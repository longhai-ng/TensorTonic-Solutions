import numpy as np

def sigmoid(x):
    """
    Vectorized sigmoid function.
    """
    # Write code here
    value = np.asarray(x, dtype = float)
    result =  1 / (1+ np.exp(-value))
    return result