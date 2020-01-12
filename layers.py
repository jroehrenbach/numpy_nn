# -*- coding: utf-8 -*-
"""
Created on Sat Jan 11 18:19:02 2020

@author: jroeh
"""

import numpy as np 
import functions as F


class Input:
    """input layer for model"""
    
    def __init__(self, neurons):
        """
        Parameters
        ----------
        neurons : int
            Number of inputs
        """
        self.neurons = neurons
    
    def feed(self, inputs):
        """
        Starts forward propagation
        
        Parameters
        ----------
        inputs : np.ndarray
            Inputs of the model
        """
        self.out = inputs
        if hasattr(self, "next"):
            self.next.forward(inputs)
 

class Layer:
    """regular layer"""
    
    def __init__(self, neurons, activation, previous, has_biases=True):
        """
        Parameters
        ----------
        neurons : int
            Number of neurons in layer
        activation : str
            Name of activation function for all neurons in layer
        previous : Layer
            Predecessor layer, for first hidden layer pass Input
        has_bias : bool
            If true, layer will have biases besides weights
        """
        self.neurons = neurons
        self.has_biases = has_biases
        self.previous = previous
        previous.next = self
        self.activation = getattr(F, activation)
        
        # set random weights and biases for layer
        self.weights = np.random.uniform(size=(previous.neurons, neurons))
        if has_biases:
            self.biases = np.random.uniform(size=(1, neurons))
    
    def forward(self, inputs, call_next=True):
        """
        Forward propagation through layer
        
        Parameters
        ----------
        inputs : np.ndarray
            Inputs that are fed to the layer
        call_next : bool
            If true, next layer will continue forward propagation
        """
        # calculate net input signal
        self.net = np.dot(inputs, self.weights) + self.biases
        # calculate activations of layer
        self.out = self.activation(self.net)
        if call_next and hasattr(self, "next"):
            self.next.forward(self.out)
    
    def backward(self, loss_gradients, lr, call_previous=True):
        """
        Backpropagation through layer and update of weights
        
        Parameters
        ----------
        loss_gradients : np.ndarray
            Gradients of loss function
        lr : float
            Learning rate, used for updating weights and biases
        call_previous : bool
            If true, backpropagation will be continued in predecessor layer. 
        """
        # calculate gradients of layer (loss gradients * activation gradients)
        self.gradients = loss_gradients * self.activation(self.net, True)
        
        # update weights
        self.weights -= self.previous.out.T.dot(self.gradients) * lr
        if self.has_biases:
            # update biases
            self.biases -= (self.gradients).sum(0, keepdims=True) * lr
        
        if call_previous and type(self.previous) != Input:
            # loss gradients for predecessor layer
            _loss_gradients = self.gradients.dot(self.weights.T)
            self.previous.backward(_loss_gradients, lr)


def test():
    i = Input(2)
    h = Layer(2, "sigmoid", i)
    o = Layer(1, "sigmoid", h)
    
    X = np.array([[0,0],[0,1],[1,0],[1,1]])
    Y = np.array([[0],[1],[1],[0]])
    lr = 0.5
    
    i.feed(X)
    print(o.out)
    
    o.backward(o.out - Y, lr)
    
    i.feed(X)
    print(o.out)


if __name__ == "__main__":
    test()