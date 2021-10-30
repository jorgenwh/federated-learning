import numpy as np
from numpynet.gate import _Gate

class _Activation(_Gate):
  def __init__(self):
    pass

  def forward(self, *args, **kwargs):
    raise NotImplementedError("forward has not been implemented for this activation function object.")

  def backward(self, *args, **kwargs):
    raise NotImplementedError("backward has not been implemented for this activation function object.")

class ReLU(_Activation):
  def __init__(self):
    self.X = None

  def __call__(self, X: np.ndarray, grad: bool = True) -> np.ndarray:
    return self.forward(X, grad)

  def forward(self, X: np.ndarray, grad: bool = True) -> np.ndarray:
    if grad:
      self.X = X
    return np.maximum(X, 0)

  def backward(self, dY: np.ndarray) -> np.ndarray:
    assert self.X is not None
    return dY * np.greater(self.X, 0).astype(int)