
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
from matplotlib.colors import ListedColormap

plt.style.use('fivethirtyeight') # this is style of graphs


class Perceptron:
  def __init__(self, eta, epochs):
    self.weights = np.random.rand(3)*1e-4
    print(f'initial weights before training = \n{self.weights}')
    self.eta = eta
    self.epochs = epochs


  def activation_function(self, inputs, weights):
    z =  np.dot(inputs, weights) #z = W * X
    return np.where(z > 0, 1, 0)

  def fit(self, X, y):
    self.X = X
    self.y = y

    X_with_bias = np.c_[self.X, -np.ones((len(self.X), 1))]
    print(f'X_with_bias = \n{X_with_bias}')

    for epoch in range(self.epochs):
      print('--'*10)
      print(f'for epoch: {epoch}')
      print('--'*10)
      
      self.y_hat = self.activation_function(X_with_bias, self.weights) # forward propogation
      print(f'predicted value after forward pass: \n{self.y_hat}')
      self.error = self.y - self.y_hat
      print(f'error: {self.error}')
      self.weights = self.weights +  self.eta * np.dot(X_with_bias.T, self.error) # backward propogation
      print(f'updated weights after epoch:\n {epoch}/{self.epochs} : {self.weights}')
      print('##'*40)



  def predict(self, X):
    X_with_bias = np.c_[X , -np.ones((len(X), 1))] 
    return self.activation_function(X_with_bias, self.weights)
  
  def total_loss(self):
    total_loss = np.sum(self.error)
    print(f'total_loss = {total_loss}')
    return total_loss