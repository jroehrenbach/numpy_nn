# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 15:04:31 2019

@author: jakob
"""

import functions as F
import layers as L
import numpy as np


class Model:
    """model which contains list of layers"""
    
    def __init__(self, input_size, loss='mean_squared_error'):
        """
        Paramters
        ---------
        input_size : int
            Size of input array
        loss : str
            Name of loss function
        """
        self.loss = getattr(F,loss)
        self.layers = [L.Input(input_size)]
    
    def add_layer(self, neurons, activation, has_biases=True, ltype="Layer"):
        """
        Adds layer to model
        
        Parameters
        ----------
        neruons : int
            Number of neurons in layer
        activation : str
            Name of activation which is used in layer
        has_biases : bool
            If true, biases will be set for layer
        ltype : str
            Type of layer
        """
        l = getattr(L, ltype)(neurons, activation, self.layers[-1], has_biases)
        self.layers.append(l)
    
    def predict(self, X):
        """
        Predicts Y by propagating through layers
        
        Parameters
        ----------
        X : np.ndarray
            Inputs of model
        """
        self.layers[0].feed(X)
        return self.layers[-1].out
    
    def train(self, X, Y, learning_rate):
        """
        Trains model using X,Y data-set
        
        Paramters
        ---------
        X, Y : np.ndarray
            Inputs of model and ground truth
        learning_rate : float
            Rate of how fast model should learn
        """
        pred = self.predict(X)
        loss_gradient = self.loss(Y, pred, True)
        self.layers[-1].backward(loss_gradient, learning_rate)


def test():
    model = Model(2, 'mean_squared_error')
    model.add_layer(2,'sigmoid')
    model.add_layer(1,'sigmoid')
    
    X = np.array([[0,0],[0,1],[1,0],[1,1]])
    Y = np.array([[0],[1],[1],[0]])
    
    lr = 0.1
    epochs = 10000
    for _ in range(epochs):
        model.train(X, Y, lr)
    print("final model prediction:")
    print(model.predict(X))


if __name__ == '__main__':
    test()