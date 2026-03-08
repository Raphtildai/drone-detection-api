# localization.py

import numpy as np

def localize_drone(mic_positions, tof):
    """
    mic_positions: np.array of shape (N,2)
    tof: time of flight array in seconds
    returns estimated position (x,y)
    """
    c = 343.0
    distances = tof * c
    # Simple centroid approx
    weights = 1 / (distances + 1e-6)
    pos = np.average(mic_positions, axis=0, weights=weights)
    return pos
