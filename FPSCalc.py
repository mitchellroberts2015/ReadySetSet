import time
import numpy as np

class FPSCalc :
  def __init__(self, len) :
      self.len = len
      self.queue = np.empty((self.len,))
      self.queue[:] = np.nan

  def frame(self, ) :
      self.queue[1:] = self.queue[:-1]
      self.queue[0] = int(round(time.time() * 1000))

  def fps(self, ) :
    mspf = (self.queue[0] - self.queue[-1]) / (self.len - 1)
    return 1000 / mspf
