import numpy as np
from numpynet.gate import _Gate

class _Loss(_Gate):
  def __init__(self):
    pass

  def forward(self, y: np.ndarray, t: np.ndarray, grad: bool = True) -> float:
    raise NotImplementedError("forward has not been implemented for this loss object.")

  def backward(self) -> np.ndarray:
    raise NotImplementedError("backward has not been implemented for this loss object.")

  def zero_grad(self) -> None:
    raise NotImplementedError("zero_grad has not been implemented for this loss object.")

class MSE(_Loss):
  def __init__(self):
    self.cache = None

  def __call__(self, y: np.ndarray, t: np.ndarray, grad: bool = True) -> float:
    return self.forward(y, t, grad)

  def forward(self, y: np.ndarray, t: np.ndarray, grad: bool = True) -> float:
    error = y - t 
    if grad:
      self.cache = error
    return np.sum(error ** 2) / y.shape[0]

  def backward(self) -> np.ndarray:
    assert self.cache is not None
    return (2 * self.cache) / self.cache.shape[0] 

  def zero_grad(self) -> None:
    self.cache = None