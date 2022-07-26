import numpy as np

def sigmoid(x: np.ndarray) -> np.ndarray:
    res = 1 / (1 + np.exp(- x))
    return res

def relu(x: np.ndarray) -> np.ndarray:
    x[x<0] = 0
    return x

def tanh(x: np.ndarray) -> np.ndarray:
    res = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
    return res
