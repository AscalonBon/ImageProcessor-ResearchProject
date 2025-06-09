from scipy import *
from sympy import *
import numpy as np

def mseCalculation():
    
    #params
    if not isinstance(I, np.ndarray) or not isinstance(K, np.ndarray):
        raise TypeError("Input and recovered images must be arrays")

    if I.shape != K.shape:
        raise ValueError("Input and recovered images must have same dimension")

    M, N = I.shape

    imageSquared_difference = (I - K)**2

    MSE = np.sum(imageSquared_difference) / (M*N) #MSE Formula
    return MSE

#array format ==Brown -- gawa ka ng dimension getter para sa input and recovered image(image after phase 2) on your side

K = np.array([
    [10, 20, 30, 40, 50],
    [15, 25, 35, 45, 55],
    [12, 22, 32, 42, 52],
    [18, 28, 38, 48, 58],
    [11, 21, 31, 41, 51]
], dtype=np.float64) #sample array of input image


I = np.array([
    [11, 21, 30, 40, 50],
    [15, 26, 35, 45, 55],
    [12, 22, 33, 42, 52],
    [18, 28, 38, 49, 58],
    [10, 21, 31, 41, 51]
], dtype=np.float64) #sample array of recovered image
    
mse_value = mseCalculation()
print(f"Calculated MSE: {mse_value}")



