# -*- coding: utf-8 -*-
"""
Created on Sat Jan 11 00:33:19 2020

@author: jroeh
"""


import numpy as np


def sigmoid(x, derive=False):
    ex = np.exp(-x)
    if derive:
        return ex / (ex+1)**2
    else:
        return 1 / (1 + ex)
    
def tanh(x, derive=False):
    if derive:
        return 1 - np.tanh(x)**2
    else:
        return np.tanh(x)

def mean_squared_error(t, p, derive=False):
    diff = p - t
    if derive:
        return 2*diff
    else:
        return diff**2