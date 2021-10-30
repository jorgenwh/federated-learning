class _Gate():
  def __init__(self, *args, **kwargs):
    pass

  def __call__(self, *args, **kwargs):
    raise NotImplementedError("__call__ has not been implemented for this gate object.")

  def forward(self, *args, **kwargs):
    raise NotImplementedError("forward has not been implemented for this gate object.")

  def backward(self, *args, **kwargs):
    raise NotImplementedError("backward has not been implemented for this gate object.")