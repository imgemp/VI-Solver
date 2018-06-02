from itertools import permutations

import numpy as np

from VISolver.Domain import Domain
from VISolver.Domains.LinearField import LinearField
from VISolver.Utilities import UnpackFlattened


class ApproxLF(LinearField):

  def __init__(self,X,dy,seqs=None,batch_size=10,eps=1e-4):
    self.X = np.reshape(X,(X.shape[0],-1))
    self.XDim = X.shape[1]
    XDim = self.XDim
    self.param_shapes = [(XDim,XDim),(XDim,)]

    self.dy = dy
    self.seqs = seqs
    if seqs is None:
      assert dy.shape == (X.shape[0],)
      seqs = list(permutations(range(X.shape[0]),2))
      dy = [dy[xf]-dy[x0] for x0,xf in seqs]
    else:
      assert dy.shape[0] == len(seqs)
    seqdy = list(zip(seqs,dy))
    np.random.shuffle(seqdy)
    self.seqdy = seqdy
    self.Nseqs = len(seqdy)

    self.batch_size = batch_size

    self.eps = eps

  '''
  Compute gradient of field prediction with respect to field parameters
  '''
  def F(self,params):
    A,b = self.ExtractParams(params)

    # Retrieve data
    idxs = np.random.choice(self.Nseqs,size=self.batch_size)

    grad = np.zeros_like(params)

    for idx in idxs:
      seq,dy = self.seqdy[idx]
      xi = [self.X[i] for i in seq]

      # Compute predictions and gradients
      dy_field = 0
      dy_field_grad = np.zeros_like(params)
      for x0,xf in zip(xi[:-1],xi[1:]):
        dy_field += self.predict(params=[A,b],t0=0,x0=x0,y0=0,tf=1,xf=xf,t=1)
        dy_field_grad += self.gradient(params,t0=0,x0=x0,y0=0,tf=1,xf=xf,t=1)

      derr = self.derror(dy_field,dy)

      grad += derr*dy_field_grad/self.batch_size

    return grad

  '''
  Compute error of field prediction
  '''
  def error(self,params):
    A,b = self.ExtractParams(params)

    # Retrieve data
    idxs = np.random.choice(self.Nseqs,size=self.batch_size)

    err = 0

    for idx in idxs:
      seq,dy = self.seqdy[idx]
      xi = [self.X[i] for i in seq]

      # Compute predictions
      dy_field = 0
      for x0,xf in zip(xi[:-1],xi[1:]):
        dy_field += self.predict(params=[A,b],t0=0,x0=x0,y0=0,tf=1,xf=xf,t=1)

      err += 0.5*(dy_field-dy)**2/self.batch_size

    return err

  '''
  Compute derivative of error
  '''
  def derror(self,dy_field,dy):
    return dy_field-dy

  ###### PARAMETER PREPARATION

  def UnpackFieldParams(self,params):
    # Unpacked like A,b
    A,b = UnpackFlattened(params,self.param_shapes)
    return A,b

  def ExtractParams(self,params):
    if len(params) == 2:
      return params
    else:
      return self.UnpackFieldParams(params)

  ###### KEY FUNCTIONS FOR LEARNING

  def predict(self,params,t0,x0,y0,tf,xf,t=None):
    if t is None:
      t = tf
    A,b = self.ExtractParams(params)
    return y0 + self.path_integral(t,t0,tf,x0,xf,A,b)

  def gradient(self,params,t0,x0,y0,tf,xf,t=None):
    if t is None:
      t = tf
    A,b = self.ExtractParams(params)

    dt = t - t0
    dt0f = tf - t0
    
    dA = np.outer(xf-x0,(x0+xf)/2)*dt/dt0f  # trapezoid rule
    # dA = np.outer(xf-x0,x0)*dt/dt0f
    # dA = np.outer(xf-x0,xf)*dt/dt0f
    db = (xf-x0)*dt/dt0f

    return np.hstack([dA.flatten(),db])


  ###### KEY PHYSICAL QUANTITIES OF INTEREST

  def path_integral(self,t,t0,tf,x0,xf,A,b):
    dt = t - t0
    dt0f = tf - t0

    Fx0 = A.dot(x0)+b
    Fxf = A.dot(xf)+b
    dx = xf-x0

    upper = Fxf.dot(dx)*dt/dt0f
    lower = Fx0.dot(dx)*dt/dt0f

    return (upper+lower)/2  # trapezoid rule
    # return upper
