"""util.py.

This file includes enumeration classe and utility functions.
"""
import numpy as np
from enum import Enum

# ----------------------------------------------------------------------------
# Enumeration Classes
class ModelInit(Enum):
    """Mode classes."""

    new = 1
    restore_train = 2
    restore_test = 3

def normalize(x):
    
    x_mean = x.mean(axis=0)
    x_stdev = x.std(axis=0)
    x_normalized = (x - x_mean) / (x_stdev + 1e-6)
    return x_normalized

def shuffle_dataset(x, y):
    
    data_size = x.shape[0]
    
    s = np.arange(data_size)
    np.random.shuffle(s)
    
    return x[s], y[s]

def split_dataset(x, y):

    data_size = x.shape[0]
    test_ratio = 0.1
    valid_ratio = 0.1
    train_ratio = 1 - test_ratio - valid_ratio

    # train set
    train_end = int(data_size*train_ratio)
    x_train = x[:train_end]
    y_train = y[:train_end]

    #  validation set
    valid_end = int(data_size*(train_ratio+valid_ratio))
    x_valid = x[train_end:valid_end]
    y_valid = y[train_end:valid_end]

    #  test set
    x_test = x[valid_end:-1]
    y_test = y[valid_end:-1]
    
    return x_train, y_train, x_valid, y_valid, x_test, y_test