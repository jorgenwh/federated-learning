import numpy as np
from typing import Tuple
from math import sqrt
from numpynet.gate import _Gate

class _Layer(_Gate):
  def __init__(self, *args, **kwargs):
    pass

  def __call__(self, *args, **kwargs):
    raise NotImplementedError("__call__ has not been implemented for this layer object.")

  def forward(self, *args, **kwargs):
    raise NotImplementedError("forward has not been implemented for this layer object.")

  def backward(self, *args, **kwargs):
    raise NotImplementedError("backward has not been implemented for this layer object.")

  def update_step(self, lr: float):
    raise NotImplementedError("update_step has not been implemented for this layer object.")

  def zero_grad(self):
    raise NotImplementedError("zero_grad has not been implemented for this layer object.")

  def get_parameters(self):
    raise NotImplementedError("get_parameters has not been implemented for this layer object.")

    
class Linear(_Layer):
  def __init__(self, in_size: int, out_size: int):
    self.in_size = in_size
    self.out_size = out_size
    
    # Initialize the parameters
    self.W = np.random.randn(in_size, out_size) / sqrt(in_size)
    self.b = np.random.randn(1, out_size) / sqrt(in_size)

    # Local variables to cache necessary data
    self.X = None
    self.output = None
    self.gradW = np.zeros((in_size, out_size))
    self.gradb = np.zeros((1, out_size))

  def __call__(self, X: np.ndarray, grad: bool = True) -> np.ndarray:
    return self.forward(X, grad)

  def forward(self, X: np.ndarray, grad: bool = True) -> np.ndarray:
    output = np.dot(X, self.W) + self.b
    if grad:
      self.X = X
      self.output = output
    return output

  def backward(self, dY: np.ndarray) -> np.ndarray:
    assert self.X is not None
    self.gradW = np.dot(self.X.T, dY)
    self.gradb = np.sum(dY, axis=0, keepdims=True)
    dX = np.dot(dY, self.W.T)
    return dX

  def update_step(self, lr: float) -> None:
    self.W -= (lr * self.gradW)
    self.b -= (lr * self.gradb)

  def zero_grad(self) -> None:
    self.gradW *= 0
    self.gradb *= 0

  def get_parameters(self) -> Tuple[np.ndarray, np.ndarray]:
    return (self.W, self.b)
