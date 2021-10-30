import numpy as np
from typing import List
from numpynet.layers import _Layer
from numpynet.losses import _Loss

class Model():
  def __init__(self, loss: _Loss):
    self.layers = []
    self.loss_function = loss
      
  def add_layer(self, layer: _Layer) -> None:
    self.layers.append(layer)

  def __call__(self, X: np.ndarray, grad: bool = True) -> np.ndarray:
    return self.forward(X, grad=grad)

  def forward(self, X: np.ndarray, grad: bool = True) -> np.ndarray:
    for layer in self.layers:
      X = layer(X, grad=grad)
    return X

  def loss(self, X: np.ndarray, t: np.ndarray, grad: bool = True) -> float:
    return self.loss_function(X, t, grad=grad)

  def backward(self) -> None:
    grad = self.loss_function.backward()
    for layer in reversed(self.layers):
      grad = layer.backward(grad)

  def update_step(self, lr: float) -> None:
    for layer in self.layers:
      if isinstance(layer, _Layer):
        layer.update_step(lr)

  def zero_grad(self) -> None:
    for layer in self.layers:
      if isinstance(layer, _Layer):
        layer.zero_grad()

  def get_parameters(self) -> List[np.ndarray]:
    parameters = []
    for layer in self.layers:
      if isinstance(layer, _Layer):
        parameters.append(layer.get_parameters())