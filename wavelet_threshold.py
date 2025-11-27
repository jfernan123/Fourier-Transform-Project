import numpy as np
import math

def calculate_threshold(cH, cV, cD):
    assert cH.shape == cV.shape == cD.shape

    M = cH.size

    # Estimate of std dev
    # https://en.wikipedia.org/wiki/Median_absolute_deviation
    sigma = np.median(np.absolute(cD)) / 0.6745
    # VisuShrink threshold
    T = sigma * math.sqrt(2 * math.log(M))

    G = np.sum(cH) + np.sum(cV) + np.sum(cD)
    S = G/M

    P = math.exp((T - S)/(T + S))

    T_new = sigma * P

    return T_new
