from itertools import product as crossprod
from itertools import combinations
import copy

import numpy as np

from VISolver.Projection import Projection
from VISolver.Domain import Domain
from VISolver.Utilities import UnpackFlattened, GramSchmidt

from IPython import embed
class LFProj(Projection):
  def __init__(self,param_shapes):
    self.param_shapes = param_shapes

  def P(self,Data,Step=0.,Direc=0.):
    # take step
    Data += Step*Direc

    # unpack data
    B,P,D,b = UnpackFlattened(Data,self.param_shapes)

    # enforce unitary constraint on P
    P = GramSchmidt(P,normalize=True)

    return np.hstack([B.flatten(),P.flatten(),D.flatten(),b.flatten()])


class LinearField(Domain):

  def __init__(self,X,dy,seqs=None,batch_size=100,eps=1e-4):
    self.X = np.reshape(X,(X.shape[0],-1))
    self.XDim = X.shape[1]
    XDim = self.XDim
    self.param_shapes = [(XDim,XDim),(XDim,XDim),(XDim,),(XDim,)]

    self.dy = dy
    self.seqs = seqs
    if seqs is None:
      assert dy.shape == (X.shape[0],)
      seqs = list(combinations(range(X.shape[0]),2))
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
    # Retrieve data
    idxs = np.random.choice(self.Nseqs,size=self.batch_size)

    grad = np.zeros_like(params)
    # grad_check = np.zeros_like(params)

    for idx in idxs:
      seq,dy = self.seqdy[idx]
      xi = [self.X[i] for i in seq]

      # Compute predictions and gradients
      dy_field = 0
      dy_field_grad = np.zeros_like(params)
      # dy_field_findiff = np.zeros_like(params)
      for x0,xf in zip(xi[:-1],xi[1:]):
        dy_field += self.predict(params,t0=0,x0=x0,y0=0,tf=1,xf=xf,t=1)
        dy_field_grad += self.gradient(params,t0=0,x0=x0,y0=0,tf=1,xf=xf,t=1)
        # dy_field_findiff += self.findiff(params,t0=0,x0=x0,y0=0,tf=1,xf=xf,t=1)

      derr = self.derror(dy_field,dy)

      grad += derr*dy_field_grad/self.batch_size
      # grad_check += derr*dy_field_findiff/self.batch_size

    return grad

  '''
  Compute error of field prediction
  '''
  def error(self,params):
    # Retrieve data
    idxs = np.random.choice(self.Nseqs,size=self.batch_size)

    err = 0

    for idx in idxs:
      seq,dy = self.seqdy[idx]
      xi = [self.X[i] for i in seq]

      # Compute predictions
      dy_field = 0
      for x0,xf in zip(xi[:-1],xi[1:]):
        dy_field += self.predict(params,t0=0,x0=x0,y0=0,tf=1,xf=xf,t=1)

      err += 0.5*(dy_field-dy)**2/self.batch_size

    return err

  '''
  Compute derivative of error
  '''
  def derror(self,dy_field,dy):
    return dy_field-dy

  ###### PARAMETER PREPARATION

  def UnpackFieldParams(self,params):
    # Unpacked like B,P,D,b
    B,P,D,b = UnpackFlattened(params,self.param_shapes)
    B = np.real(B)
    b = np.real(b)
    D.real = np.zeros_like(D)
    Pinv = np.linalg.pinv(P)
    C = self.C(Pinv,P,D)
    A = self.A(B,C=C)
    return A,B,C,Pinv,D,P,b

  def DecomposeField(self,A,b):
    B = (A+A.T)/2
    C = (A-A.T)/2
    D,Pinv = np.linalg.eig(C)
    D.real = np.zeros_like(D)
    P = np.linalg.pinv(Pinv)
    return A,B,C,Pinv,D,P,b

  def AuxParams(self,x0,xf,A,B,C,Pinv,D,P,b):
    sqrtD = np.sqrt(D)
    Px0 = P.dot(x0)
    Pxf = P.dot(xf)
    d = self.d(P,A,b,x0)
    return sqrtD,Px0,Pxf,d

  def ExtractParams(self,params):
    if len(params) == 2:
      return self.DecomposeField(*params)
    else:
      return self.UnpackFieldParams(params)

  def Ab_to_BPDb(self,A,b,flat=False):
    B = (A+A.T)/2
    C = (A-A.T)/2
    D,Pinv = np.linalg.eig(C)
    D.real = np.zeros_like(D)
    P = np.linalg.pinv(Pinv)
    if flat:
      return np.hstack([B.flatten(),P.flatten(),D.flatten(),b.flatten()])
    else:
      return B,P,D,b

  def A(self,B,Pinv=None,P=None,D=None,C=None):
    if C is not None:
      return np.real(B + C)
    else:
      return np.real(B + self.C(Pinv,P,D))

  def C(self,Pinv,P,D):
    return np.real(Pinv.dot(np.diag(D)).dot(P))

  def d(self,P,A,b,x0):
    return -P.dot(A.T.dot(x0)+b)

  ###### KEY FUNCTIONS FOR LEARNING

  def predict(self,params,t0,x0,y0,tf,xf,t=None):
    if t is None:
      t = tf
    A,B,C,Pinv,D,P,b = self.ExtractParams(params)
    sqrtD,Px0,Pxf,d = self.AuxParams(x0,xf,A,B,C,Pinv,D,P,b)
    return y0 + self.path_integral(t,t0,tf,x0,xf,A,Pinv,P,Px0,Pxf,D,sqrtD,d,b)

  def findiff(self,params,t0,x0,y0,tf,xf,t=None,eps=1e-4):
    if t is None:
      t = tf
    A,B,C,Pinv,D,P,b = self.ExtractParams(params)
    sqrtD,Px0,Pxf,d = self.AuxParams(x0,xf,A,B,C,Pinv,D,P,b)

    fx = self.path_integral(t,t0,tf,x0,xf,A,Pinv,P,Px0,Pxf,D,sqrtD,d,b)

    fdB = np.zeros_like(B)
    for u,v in crossprod(*([range(self.XDim)]*2)):
      Btemp = copy.copy(B)
      Btemp[u,v] += eps
      Atemp = self.A(B=Btemp,C=C)
      dtemp = self.d(P,Atemp,b,x0)
      fx_h = self.path_integral(t,t0,tf,x0,xf,Atemp,Pinv,P,Px0,Pxf,D,sqrtD,dtemp,b)
      fdB[u,v] = (fx_h-fx)/eps

    fdP = np.zeros_like(P)
    for u,v in crossprod(*([range(self.XDim)]*2)):
      Ptemp = copy.copy(P)
      Ptemp[u,v] += eps
      Px0temp = Ptemp.dot(x0)
      Pxftemp = Ptemp.dot(xf)
      Pinvtemp = np.linalg.pinv(Ptemp)
      Atemp = self.A(B=B,Pinv=Pinvtemp,P=Ptemp,D=D)
      dtemp = self.d(Ptemp,Atemp,b,x0)
      fx_h = self.path_integral(t,t0,tf,x0,xf,Atemp,Pinvtemp,Ptemp,Px0temp,Pxftemp,D,sqrtD,dtemp,b)
      fdP[u,v] = (fx_h-fx)/eps

    fdD = np.zeros_like(D)
    for u in crossprod(range(self.XDim)):
      Dtemp = copy.copy(D)
      Dtemp[u] += eps*1j
      sqrtDtemp = np.sqrt(Dtemp)
      Atemp = self.A(B=B,Pinv=Pinv,P=P,D=Dtemp)
      dtemp = self.d(P,Atemp,b,x0)
      fx_h = self.path_integral(t,t0,tf,x0,xf,Atemp,Pinv,P,Px0,Pxf,Dtemp,sqrtDtemp,dtemp,b)
      fdD[u] = (fx_h-fx)/eps

    fdb = np.zeros_like(b)
    for u in crossprod(range(self.XDim)):
      btemp = copy.copy(b)
      btemp[u] += eps
      dtemp = self.d(P,A,btemp,x0)
      fx_h = self.path_integral(t,t0,tf,x0,xf,A,Pinv,P,Px0,Pxf,D,sqrtD,dtemp,btemp)
      fdb[u] = (fx_h-fx)/eps

    fd = np.hstack([fdB.flatten(),fdP.flatten(),fdD.flatten(),fdb.flatten()])

    if np.any(np.imag(fd) > self.eps):
      # Note: finite difference should always be real because path integral should always be real,
      # however, P and D are typically complex so imaginary dP and dD are expected in actual gradient
      raise ValueError('Imaginary part of finite difference gradient is suspicially large. Should always be zero.\n%g'%np.argmax(np.abs(np.imag(fd))))

    return fd

  def gradient(self,params,t0,x0,y0,tf,xf,t=None):
    if t is None:
      t = tf
    A,B,C,Pinv,D,P,b = self.ExtractParams(params)
    sqrtD,Px0,Pxf,d = self.AuxParams(x0,xf,A,B,C,Pinv,D,P,b)

    dt = t - t0
    dt0f = tf - t0
    zt = self.z(dt,dt0f,d,D,sqrtD,Px0,Pxf)
    xt = self.ztox(Pinv,zt)

    dB = np.zeros((self.XDim,self.XDim),dtype=np.complex)
    dP = np.zeros((self.XDim,self.XDim),dtype=np.complex)
    dD = np.zeros(self.XDim,dtype=np.complex)
    db = np.cast[np.complex](xf-x0)

    int_dints = self.int_dint_zzdot(A,B,Pinv,P,D,b,dt,dt0f,zt,xt,x0,xf,sqrtD,Px0,Pxf,d)
    int_zzdot, dint_zzdot_db, dint_zzdot_dD, dint_zzdot_dB, dint_zzdot_dP = int_dints

    for u,i,j,k,l in crossprod(*([range(self.XDim)]*5)):
        db[u] += A[i,j]*Pinv[j,k]*Pinv[i,l]*dint_zzdot_db[u,k,l]
        dD[u] += self.dCij_dDu(i,j,u,Pinv,P)*Pinv[j,k]*Pinv[i,l]*int_zzdot[k,l] + \
          A[i,j]*Pinv[j,k]*Pinv[i,l]*dint_zzdot_dD[u,k,l]

    for u,v,k,l in crossprod(*([range(self.XDim)]*4)):
        dB[u,v] += Pinv[v,k]*Pinv[u,l]*int_zzdot[k,l]

    for u,v,i,j,k,l in crossprod(*([range(self.XDim)]*6)):
        dB[u,v] += A[i,j]*Pinv[j,k]*Pinv[i,l]*dint_zzdot_dB[u,v,k,l]
        dP[u,v] += self.dCij_dPuv(i,j,u,v,Pinv,P,D)*Pinv[j,k]*Pinv[i,l]*int_zzdot[k,l] - \
          A[i,j]*Pinv[j,u]*Pinv[v,k]*Pinv[i,l]*int_zzdot[k,l] - \
          A[i,j]*Pinv[j,k]*Pinv[i,u]*Pinv[v,l]*int_zzdot[k,l] + \
          A[i,j]*Pinv[j,k]*Pinv[i,l]*dint_zzdot_dP[u,v,k,l]

    # Enforce constraints on gradients (dB is real-symmetric, db is real, dD is imaginary)
    # Unitary constraint on P will be enforced with Gram Schmidt "projection"
    dB = self.Symmetrize(dB)
    db = np.real(db)
    dD = self.AverageConjugatePairs(D,dD)

    dField = np.hstack([dB.flatten(),dP.flatten(),dD.flatten(),db.flatten()])

    return dField

  def Symmetrize(self,dB):
    dB = np.real(dB)
    dB = dB.reshape(self.XDim,self.XDim)
    return np.ravel((dB+dB.T)/2)

  def AverageConjugatePairs(self,D,dD):
    dD.real = np.zeros_like(dD)
    i = 0
    while i < D.shape[0] - 1:
      if np.allclose(D[i],-D[i+1]):
        dD[i] = (dD[i]-dD[i+1])/2
        dD[i+1] = -dD[i]
        i += 2
      else:
        i += 1
    return dD

  ###### KEY PHYSICAL QUANTITIES OF INTEREST

  def Field(self,params,x):
    assert x.shape[-1] == self.XDim
    A,B,C,Pinv,D,P,b = self.ExtractParams(params)
    return  A.dot(x.T).T+b

  def Action(self,t,t0,x0,y0,tf,xf,params):
    A,B,C,Pinv,D,P,b = self.ExtractParams(params)
    sqrtD,Px0,Pxf,d = self.AuxParams(x0,xf,A,B,C,Pinv,D,P,b)

    dt = t - t0
    dt0f = tf - t0
    zt = self.z(dt,dt0f,d,D,sqrtD,Px0,Pxf)
    xt = self.ztox(Pinv,zt)

    action = -np.dot(b,xt-x0)*dt - y0*dt

    for i,k,l in crossprod(*([range(self.XDim)]*3)):

      Px0k, Px0l = Px0[k], Px0[l]
      Pxfk, Pxfl = Pxf[k], Pxf[l]
      dk, dl = d[k], d[l]
      Dk, Dl = D[k], D[l]
      sqrtDk, sqrtDl = np.sqrt(Dk), np.sqrt(Dl)

      kc = np.abs(Dk) < self.eps
      lc = np.abs(Dl) < self.eps

      if kc:
        c1k = self.c1i_0(dt0f,Px0k,Pxfk,dk)
        c2k = self.c2i_0(Px0k)
      else:
        c1k = self.c1i_n0(dt0f,Px0k,Pxfk,Dk,sqrtDk,dk)
        c2k = self.c2i_n0(dt0f,Px0k,Pxfk,Dk,sqrtDk,dk)

      if lc:
        c1l = self.c1i_0(dt0f,Px0l,Pxfl,dl)
        c2l = self.c2i_0(Px0l)
      else:
        c1l = self.c1i_n0(dt0f,Px0l,Pxfl,Dl,sqrtDl,dl)
        c2l = self.c2i_n0(dt0f,Px0l,Pxfl,Dl,sqrtDl,dl)

      if kc and lc:
        int_zdotzdot = self.int_z0dot_z0dot(dt,dk,dl,c1k,c1l)
      elif kc and not lc:
        int_zdotzdot = self.int_z0dot_zn0dot(dt,dk,sqrtDl,c1l,c1k,c2l)
      elif not kc and lc:
        int_zdotzdot = self.int_zn0dot_z0dot(dt,dl,sqrtDk,c1k,c1l,c2k)
      elif not kc and not lc:
        int_zdotzdot = self.int_zn0dot_zn0dot(dt,sqrtDk,sqrtDl,c1k,c1l,c2k,c2l)

      action += 0.5*Pinv[i,k]*Pinv[i,l]*int_zdotzdot

    for i,j,k,l in crossprod(*([range(self.XDim)]*4)):

      Px0k, Px0l = Px0[k], Px0[l]
      Pxfk, Pxfl = Pxf[k], Pxf[l]
      dk, dl = d[k], d[l]
      Dk, Dl = D[k], D[l]
      sqrtDk, sqrtDl = np.sqrt(Dk), np.sqrt(Dl)

      kc = np.abs(Dk) < self.eps
      lc = np.abs(Dl) < self.eps

      if kc:
        c1k = self.c1i_0(dt0f,Px0k,Pxfk,dk)
        c2k = self.c2i_0(Px0k)
      else:
        c1k = self.c1i_n0(dt0f,Px0k,Pxfk,Dk,sqrtDk,dk)
        c2k = self.c2i_n0(dt0f,Px0k,Pxfk,Dk,sqrtDk,dk)

      if lc:
        c1l = self.c1i_0(dt0f,Px0l,Pxfl,dl)
        c2l = self.c2i_0(Px0l)
      else:
        c1l = self.c1i_n0(dt0f,Px0l,Pxfl,Dl,sqrtDl,dl)
        c2l = self.c2i_n0(dt0f,Px0l,Pxfl,Dl,sqrtDl,dl)

      if kc and lc:
        intint_zzdot = self.intint_z0_z0dot(dt,dk,dl,c1k,c1l,c2k,c2l)
      elif kc and not lc:
        intint_zzdot = self.intint_z0_zn0dot(dt,dk,Dl,sqrtDl,c1k,c1l,c2k,c2l)
      elif not kc and lc:
        intint_zzdot = self.intint_zn0_z0dot(dt,dk,dl,Dk,sqrtDk,c1k,c1l,c2k)
      elif not kc and not lc:
        intint_zzdot = self.intint_zn0_zn0dot(dt,dk,Dk,sqrtDk,sqrtDl,c1k,c1l,c2k,c2l)

      action -= A[i,j]*Pinv[j,k]*Pinv[i,l]*intint_zzdot

    if np.imag(action) > self.eps:
      raise ValueError('Imaginary part of action is suspicially large. Should always be zero.\n%g'%np.imag(action))

    return np.real(action)

  def Lagrangian(self,t,t0,x0,y0,tf,xf,params):
    A,B,C,Pinv,D,P,b = self.ExtractParams(params)
    sqrtD,Px0,Pxf,d = self.AuxParams(x0,xf,A,B,C,Pinv,D,P,b)
    xdots = self.xdot(t,t0,tf,d,D,sqrtD,Px0,Pxf,Pinv)
    L = 0.5*np.dot(xdots,xdots) - self.predict(params,t0,x0,y0,tf,xf,t)
    if np.imag(L) > self.eps:
      raise ValueError('Imaginary part of Lagrangian is suspicially large. Should always be zero.\n%g'%np.imag(L))
    return np.real(L)

  def EulerLagrange(self,t,t0,x0,y0,tf,xf,params):
    A,B,C,Pinv,D,P,b = self.ExtractParams(params)
    sqrtD,Px0,Pxf,d = self.AuxParams(x0,xf,A,B,C,Pinv,D,P,b)
    assert np.allclose(A-A.T,2*C)

    z = self.z(t-t0,tf-t0,d,D,sqrtD,Px0,Pxf)
    zddot = self.zddot(t-t0,tf-t0,d,D,sqrtD,Px0,Pxf)
    ELz = zddot.T - (z.T*D) + d
    EL = Pinv.dot(ELz.T).T

    # Lose numerical precision when multiplying with Pinv many times
    # Computation above in zspace is more stable
    # x = self.x(t,t0,tf,x0,xf,params)
    # xddot = self.xddot(t,t0,tf,d,D,sqrtD,Px0,Pxf,Pinv)
    # EL = xddot.T - 2*(C.dot(x.T)).T + A.T.dot(x0)+b

    if np.any(np.abs(np.imag(EL)) > self.eps):
      raise ValueError('Imaginary part of Euler-Lagrange equations is suspicially large. Should always be zero.\n%g'%np.imag(EL))

    return np.real(EL)

  def path_integral(self,t,t0,tf,x0,xf,A,Pinv,P,Px0,Pxf,D,sqrtD,d,b):
    dt = t - t0
    dt0f = tf - t0

    zt = self.z(dt,dt0f,d,D,sqrtD,Px0,Pxf)
    xt = self.ztox(Pinv,zt)

    int_zzdot = np.zeros((self.XDim,self.XDim),dtype=np.complex)
    for k,l in crossprod(*([range(self.XDim)]*2)):
      Px0k, Px0l = Px0[k], Px0[l]
      Pxfk, Pxfl = Pxf[k], Pxf[l]
      dk, dl = d[k], d[l]
      Dk, Dl = D[k], D[l]
      sqrtDk, sqrtDl = np.sqrt(Dk), np.sqrt(Dl)

      kc = np.abs(Dk) < self.eps
      lc = np.abs(Dl) < self.eps

      if kc:
        # Dk = 0j
        c1k = self.c1i_0(dt0f,Px0k,Pxfk,dk)
        c2k = self.c2i_0(Px0k)
      else:
        c1k = self.c1i_n0(dt0f,Px0k,Pxfk,Dk,sqrtDk,dk)
        c2k = self.c2i_n0(dt0f,Px0k,Pxfk,Dk,sqrtDk,dk)

      if lc:
        # Dl = 0j
        c1l = self.c1i_0(dt0f,Px0l,Pxfl,dl)
        c2l = self.c2i_0(Px0l)
      else:
        c1l = self.c1i_n0(dt0f,Px0l,Pxfl,Dl,sqrtDl,dl)
        c2l = self.c2i_n0(dt0f,Px0l,Pxfl,Dl,sqrtDl,dl)

      if kc and lc:
        int_zzdot[k,l] = self.int_z0_z0dot(dt,dk,dl,c1k,c1l,c2k,c2l)
      elif kc and not lc:
        int_zzdot[k,l] = self.int_z0_zn0dot(dt,dk,Dl,sqrtDl,c1k,c1l,c2k,c2l)
      elif not kc and lc:
        int_zzdot[k,l] = self.int_zn0_z0dot(dt,dk,dl,Dk,sqrtDk,c1k,c1l,c2k,c2l)
      elif not kc and not lc:
        int_zzdot[k,l] = self.int_zn0_zn0dot(dt,dk,dl,Dk,Dl,sqrtDk,sqrtDl,c1k,c1l,c2k,c2l)

    path_int = np.dot(b,xt-x0)
    for i,j,k,l in crossprod(*([range(self.XDim)]*4)):
      path_int += A[i,j]*Pinv[j,k]*Pinv[i,l]*int_zzdot[k,l]

    # if np.imag(path_int) > self.eps:
    #   raise ValueError('Imaginary part of path integral is suspicially large. Should always be zero.\n%g'%np.imag(path_int))

    return np.real(path_int)

  ###### PATHS AND DERIVATIVES

  def x(self,t,t0,tf,x0,xf,params):
    A,B,C,Pinv,D,P,b = self.ExtractParams(params)
    sqrtD,Px0,Pxf,d = self.AuxParams(x0,xf,A,B,C,Pinv,D,P,b)
    if isinstance(t,np.ndarray) or isinstance(t,list):
      return np.asarray([self.ztox(Pinv,self.z(ti-t0,tf-t0,d,D,sqrtD,Px0,Pxf)) for ti in t])
    else:
      return self.ztox(Pinv,self.z(t-t0,tf-t0,d,D,sqrtD,Px0,Pxf))

  def xdot(self,t,t0,tf,d,D,sqrtD,Px0,Pxf,Pinv):
    return self.ztox(Pinv,self.zdot(t-t0,tf-t0,d,D,sqrtD,Px0,Pxf))

  def xddot(self,t,t0,tf,d,D,sqrtD,Px0,Pxf,Pinv):
    return self.ztox(Pinv,self.zddot(t-t0,tf-t0,d,D,sqrtD,Px0,Pxf))

  def xxdot(self,j,i,Pinv,zzdot):
    Pinv_kl = np.outer(Pinv[j,k],Pinv[i,l])
    return np.sum(Pinv_kl*zzdot)

  def ztox(self,Pinv,z):
    return Pinv.dot(z)

  def z(self,dt,dt0f,d,D,sqrtD,Px0,Pxf):
    zis = []
    for i in range(self.XDim):
      Px0i = Px0[i]
      Pxfi = Pxf[i]
      Di = D[i]
      sqrtDi = sqrtD[i]
      di = d[i]
      if np.abs(D[i]) < self.eps:
        c1i = self.c1i_0(dt0f,Px0i,Pxfi,di)
        c2i = self.c2i_0(Px0i)
        zi = self.zi_0(dt,di,c1i,c2i)
      else:
        c1i = self.c1i_n0(dt0f,Px0i,Pxfi,Di,sqrtDi,di)
        c2i = self.c2i_n0(dt0f,Px0i,Pxfi,Di,sqrtDi,di)
        zi = self.zi_n0(dt,di,Di,sqrtDi,c1i,c2i)
      zis += [zi]
    return np.asarray(zis)

  def zdot(self,dt,dt0f,d,D,sqrtD,Px0,Pxf):
    zidots = []
    for i in range(self.XDim):
      Px0i = Px0[i]
      Pxfi = Pxf[i]
      Di = D[i]
      sqrtDi = sqrtD[i]
      di = d[i]
      if np.abs(D[i]) < self.eps:
        c1i = self.c1i_0(dt0f,Px0i,Pxfi,di)
        zidot = self.zidot_0(dt,di,c1i)
      else:
        c1i = self.c1i_n0(dt0f,Px0i,Pxfi,Di,sqrtDi,di)
        c2i = self.c2i_n0(dt0f,Px0i,Pxfi,Di,sqrtDi,di)
        zidot = self.zidot_n0(dt,sqrtDi,c1i,c2i)
      zidots += [zidot]
    return np.asarray(zidots)

  def zddot(self,dt,dt0f,d,D,sqrtD,Px0,Pxf):
    ziddots = []
    for i in range(self.XDim):
      Px0i = Px0[i]
      Pxfi = Pxf[i]
      Di = D[i]
      sqrtDi = sqrtD[i]
      di = d[i]
      if np.abs(D[i]) < self.eps:
        ziddot = self.ziddot_0(di)
      else:
        c1i = self.c1i_n0(dt0f,Px0i,Pxfi,Di,sqrtDi,di)
        c2i = self.c2i_n0(dt0f,Px0i,Pxfi,Di,sqrtDi,di)
        ziddot = self.ziddot_n0(dt,sqrtDi,Di,c1i,c2i)
      ziddots += [ziddot]
    return np.asarray(ziddots)

  ###### E-L SOLUTIONS (|Dii| < eps)

  def zi_0(self,dt,di,c1i,c2i):
    return -di/2*dt**2 + c1i*dt + c2i

  def zidot_0(self,dt,di,c1i):
    return -di*dt + c1i

  def ziddot_0(self,di):
    return -di

  def c1i_0(self,dt,Px0i,Pxfi,di):
    return (Pxfi-Px0i)/dt + di*dt/2

  def c2i_0(self,Px0i):
    return Px0i

  ###### E-L SOLUTIONS (|Dii| >= eps)

  def zi_n0(self,dt,di,Di,sqrtDi,c1i,c2i):
    sqrtDit = sqrtDi*dt
    return di/Di + c1i*np.exp(sqrtDit) + c2i*np.exp(-sqrtDit)

  def zidot_n0(self,dt,sqrtDi,c1i,c2i):
    sqrtDit = sqrtDi*dt
    return sqrtDi*(c1i*np.exp(sqrtDit) - c2i*np.exp(-sqrtDit))

  def ziddot_n0(self,dt,sqrtDi,Di,c1i,c2i):
    sqrtDit = sqrtDi*dt
    return Di*(c1i*np.exp(sqrtDit) + c2i*np.exp(-sqrtDit))

  def c1i_n0(self,dt,Px0i,Pxfi,Di,sqrtDi,di):
    return Px0i - di/Di - self.c2i_n0(dt,Px0i,Pxfi,Di,sqrtDi,di)

  def c2i_n0(self,dt,Px0i,Pxfi,Di,sqrtDi,di):
    sqrtDit = sqrtDi*dt
    numer = (Pxfi-di/Di) - np.exp(sqrtDit)*(Px0i-di/Di)
    denom = np.exp(-sqrtDit) - np.exp(sqrtDit)
    return numer/denom

  ###### Z-ZDOT PRODUCTS

  def z0_z0dot(self,dt,dk,dl,c1k,c1l,c2k):
    zk = self.zi_0(dt,dk,c1k,c2k)
    zldot = self.zidot_0(dt,dl,c1l)
    return zk*zldot

  def z0_z0dot_p(self,dt,dk,dl,c1k,c1l,c2k):

    zzdot_kl_1 = dk*dl/2*dt**3
    zzdot_kl_2 = -c1l*dk/2*dt**2
    zzdot_kl_3 = -c1k*dl*dt**2
    zzdot_kl_4 = c1k*c1l*dt
    zzdot_kl_5 = -c2k*dl*dt
    zzdot_kl_6 = c2k*c1l

    return np.asarray([zzdot_kl_1,zzdot_kl_2,zzdot_kl_3,
                       zzdot_kl_4,zzdot_kl_5,zzdot_kl_6])

  def z0_zn0dot(self,dt,dk,sqrtDl,c1k,c1l,c2k,c2l):
    zk = self.zi_0(dt,dk,c1k,c2k)
    zldot = self.zidot_n0(dt,sqrtDl,c1l,c2l)
    return zk*zldot

  def z0_zn0dot_p(self,dt,dk,sqrtDl,c1k,c1l,c2k,c2l):
    sqrtDlt = sqrtDl*dt

    zzdot_kl_1 = -dk*c1l*sqrtDl/2*dt**2*np.exp(sqrtDlt)
    zzdot_kl_2 = dk*c2l*sqrtDl/2*dt**2*np.exp(-sqrtDlt)
    zzdot_kl_3 = c1k*c1l*sqrtDl*dt*np.exp(sqrtDlt)
    zzdot_kl_4 = -c1k*c2l*sqrtDl*dt*np.exp(-sqrtDlt)
    zzdot_kl_5 = c2k*c1l*sqrtDl*np.exp(sqrtDlt)
    zzdot_kl_6 = -c2k*c2l*sqrtDl*np.exp(-sqrtDlt)

    return np.asarray([zzdot_kl_1,zzdot_kl_2,zzdot_kl_3,
                       zzdot_kl_4,zzdot_kl_5,zzdot_kl_6])

  def zn0_z0dot(self,dt,dk,dl,Dk,sqrtDk,c1k,c1l,c2k):
    zk = self.zi_n0(dt,dk,Dk,sqrtDk,c1k,c2k)
    zldot = self.zidot_0(dt,dl,c1l)
    return zk*zldot

  def zn0_z0dot_p(self,dt,dk,dl,Dk,c1k,c1l,c2k,c2l):
    sqrtDkt = sqrtDk*dt

    zzdot_kl_1 = -dk*dl/Dk*dt
    zzdot_kl_2 = dk*c1l/Dk
    zzdot_kl_3 = -c1k*dl*dt*np.exp(sqrtDkt)
    zzdot_kl_4 = c1k*c1l*np.exp(sqrtDkt)
    zzdot_kl_5 = -c2k*dl*dt*np.exp(-sqrtDkt)
    zzdot_kl_6 = c2k*c2l*np.exp(-sqrtDkt)

    return np.asarray([zzdot_kl_1,zzdot_kl_2,zzdot_kl_3,
                       zzdot_kl_4,zzdot_kl_5,zzdot_kl_6])

  def zn0_zn0dot(self,dt,dk,Dk,sqrtDk,sqrtDl,c1k,c1l,c2k,c2l):
    sqrtDkt = sqrtDk*dt
    zk = dk/Dk + c1k*np.exp(sqrtDkt) + c2k*np.exp(-sqrtDkt)
    sqrtDlt = sqrtDl*dt
    zldot = sqrtDl*(c1l*np.exp(sqrtDlt) - c2l*np.exp(-sqrtDlt))
    return zk*zldot

  def zn0_zn0dot_p(self,dt,dk,Dk,sqrtDk,sqrtDl,c1k,c1l,c2k,c2l):
    sqrtDkt = sqrtDk*dt
    sqrtDlt = sqrtDl*dt

    zzdot_kl_1 = dk/Dk*c1l*sqrtDl*np.exp(sqrtDlt)
    zzdot_kl_2 = -dk/Dk*c2l*sqrtDl*np.exp(-sqrtDlt)
    zzdot_kl_3 = c1k*c1l*sqrtDl*np.exp(sqrtDkt+sqrtDlt)
    zzdot_kl_4 = -c1k*c2l*sqrtDl*np.exp(sqrtDkt-sqrtDlt)
    zzdot_kl_5 = c2k*c1l*sqrtDl*np.exp(-sqrtDkt+sqrtDlt)
    zzdot_kl_6 = -c2k*c2l*sqrtDl*np.exp(-sqrtDkt-sqrtDlt)

    return np.asarray([zzdot_kl_1,zzdot_kl_2,zzdot_kl_3,
                       zzdot_kl_4,zzdot_kl_5,zzdot_kl_6])

  ###### Z-ZDOT INTEGRALS

  def int_z0_z0dot(self,dt,dk,dl,c1k,c1l,c2k,c2l):
    return np.sum(self.int_z0_z0dot_p(dt,dk,dl,c1k,c1l,c2k,c2l))

  def int_z0_z0dot_p(self,dt,dk,dl,c1k,c1l,c2k,c2l):

    intzzdot_kl_1 = dk*dl/8*dt**4
    intzzdot_kl_2 = -c1l*dk/6*dt**3
    intzzdot_kl_3 = -c1k*dl/3*dt**3
    intzzdot_kl_4 = c1k*c1l/2*dt**2
    intzzdot_kl_5 = -c2k*dl/2*dt**2
    intzzdot_kl_6 = c2k*c1l*dt

    return np.asarray([intzzdot_kl_1,intzzdot_kl_2,intzzdot_kl_3,
                       intzzdot_kl_4,intzzdot_kl_5,intzzdot_kl_6])

  def int_z0_zn0dot(self,dt,dk,Dl,sqrtDl,c1k,c1l,c2k,c2l):
    return np.sum(self.int_z0_zn0dot_p(dt,dk,Dl,sqrtDl,c1k,c1l,c2k,c2l))

  def int_z0_zn0dot_p(self,dt,dk,Dl,sqrtDl,c1k,c1l,c2k,c2l):
    sqrtDlt = sqrtDl*dt

    intzzdot_kl_1 = -dk*c1l/(2*Dl)*(np.exp(sqrtDlt)*((sqrtDlt-1)**2+1)-2)
    intzzdot_kl_2 = -dk*c2l/(2*Dl)*(np.exp(-sqrtDlt)*((-sqrtDlt-1)**2+1)-2)
    intzzdot_kl_3 = c1k*c1l/sqrtDl*(np.exp(sqrtDlt)*(sqrtDlt-1)+1)
    intzzdot_kl_4 = -c1k*c2l/sqrtDl*(np.exp(-sqrtDlt)*(-sqrtDlt-1)+1)
    intzzdot_kl_5 = c2k*c1l*(np.exp(sqrtDlt)-1)
    intzzdot_kl_6 = c2k*c2l*(np.exp(-sqrtDlt)-1)

    return np.asarray([intzzdot_kl_1,intzzdot_kl_2,intzzdot_kl_3,
                       intzzdot_kl_4,intzzdot_kl_5,intzzdot_kl_6])

  def int_zn0_z0dot(self,dt,dk,dl,Dk,sqrtDk,c1k,c1l,c2k,c2l):
    return np.sum(self.int_zn0_z0dot_p(dt,dk,dl,Dk,sqrtDk,c1k,c1l,c2k,c2l))

  def int_zn0_z0dot_p(self,dt,dk,dl,Dk,sqrtDk,c1k,c1l,c2k,c2l):
    sqrtDkt = sqrtDk*dt

    intzzdot_kl_1 = -dk*dl/(2*Dk)*dt**2
    intzzdot_kl_2 = dk*c1l/Dk*dt
    intzzdot_kl_3 = -c1k*dl/Dk*(np.exp(sqrtDkt)*(sqrtDkt-1)+1)
    intzzdot_kl_4 = c1k*c1l/sqrtDk*(np.exp(sqrtDkt)-1)
    intzzdot_kl_5 = -c2k*dl/Dk*(np.exp(-sqrtDkt)*(-sqrtDkt-1)+1)
    intzzdot_kl_6 = -c2k*c1l/sqrtDk*(np.exp(-sqrtDkt)-1)

    return np.asarray([intzzdot_kl_1,intzzdot_kl_2,intzzdot_kl_3,
                       intzzdot_kl_4,intzzdot_kl_5,intzzdot_kl_6])

  def int_zn0_zn0dot(self,dt,dk,dl,Dk,Dl,sqrtDk,sqrtDl,c1k,c1l,c2k,c2l):
    return np.sum(self.int_zn0_zn0dot_p(dt,dk,dl,Dk,Dl,sqrtDk,sqrtDl,c1k,c1l,c2k,c2l))

  def int_zn0_zn0dot_p(self,dt,dk,dl,Dk,Dl,sqrtDk,sqrtDl,c1k,c1l,c2k,c2l):
    sqrtDkt = sqrtDk*dt
    sqrtDlt = sqrtDl*dt

    intzzdot_kl_1 = dk/Dk*c1l*(np.exp(sqrtDlt)-1)
    intzzdot_kl_2 = dk/Dk*c2l*(np.exp(-sqrtDlt)-1)
    intzzdot_kl_3 = c1k*c1l*sqrtDl/(sqrtDk+sqrtDl)*(np.exp(sqrtDkt+sqrtDlt)-1)
    if np.allclose(sqrtDk,sqrtDl):
      intzzdot_kl_4 = -c1k*c2l*sqrtDlt
      intzzdot_kl_5 = c2k*c1l*sqrtDlt
    else:
      intzzdot_kl_4 = -c1k*c2l*sqrtDl/(sqrtDk-sqrtDl)*(np.exp(sqrtDkt-sqrtDlt)-1)
      intzzdot_kl_5 = -c2k*c1l*sqrtDl/(sqrtDk-sqrtDl)*(np.exp(-sqrtDkt+sqrtDlt)-1)
    intzzdot_kl_6 = c2k*c2l*sqrtDl/(sqrtDk+sqrtDl)*(np.exp(-sqrtDkt-sqrtDlt)-1)

    return np.asarray([intzzdot_kl_1,intzzdot_kl_2,intzzdot_kl_3,
                       intzzdot_kl_4,intzzdot_kl_5,intzzdot_kl_6],dtype=np.complex)

  ###### ZDOT-ZDOT PRODUCTS

  def z0dot_z0dot(self,dt,dk,dl,c1k,c1l):
    zkdot = self.zidot_0(dt,dk,c1k)
    zldot = self.zidot_0(dt,dl,c1l)
    return zkdot*zldot

  def z0dot_z0dot_p(self,dt,dk,dl,c1k,c1l):

    zdotzdot_kl_1 = dk*dl*dt**2
    zdotzdot_kl_2 = -c1l*dk*dt
    zdotzdot_kl_3 = -c1k*dl*dt
    zdotzdot_kl_4 = c1k*c1l

    return np.asarray([zdotzdot_kl_1,zdotzdot_kl_2,zdotzdot_kl_3,zdotzdot_kl_4])

  def z0dot_zn0dot(self,dt,dk,sqrtDl,c1k,c1l,c2l):
    zkdot = self.zidot_0(dt,dk,c1k)
    zldot = self.zidot_n0(dt,sqrtDl,c1l,c2l)
    return zkdot*zldot

  def z0dot_zn0dot_p(self,dt,dk,sqrtDl,c1k,c1l,c2l):
    sqrtDlt = sqrtDl*dt

    zdotzdot_kl_1 = -c1l*dk*sqrtDl*dt*np.exp(sqrtDlt)
    zdotzdot_kl_2 = c2l*dk*sqrtDl*dt*np.exp(-sqrtDlt)
    zdotzdot_kl_3 = c1l*c1k*sqrtDl*np.exp(sqrtDlt)
    zdotzdot_kl_4 = -c2l*c1k*sqrtDl*np.exp(-sqrtDlt)

    return np.asarray([zdotzdot_kl_1,zdotzdot_kl_2,zdotzdot_kl_3,zdotzdot_kl_4])

  def zn0dot_z0dot(self,dt,dl,sqrtDk,c1l,c1k,c2k):
    zkdot = self.zidot_n0(dt,sqrtDk,c1k,c2k)
    zldot = self.zidot_0(dt,dl,c1l)
    return zkdot*zldot

  def zn0dot_z0dot_p(self,dt,dl,sqrtDk,c1l,c1k,c2k):
    sqrtDkt = sqrtDk*dt

    zdotzdot_kl_1 = -c1k*dl*sqrtDk*dt*np.exp(sqrtDkt)
    zdotzdot_kl_2 = c2k*dl*sqrtDk*dt*np.exp(-sqrtDkt)
    zdotzdot_kl_3 = c1k*c1l*sqrtDk*np.exp(sqrtDkt)
    zdotzdot_kl_4 = -c2k*c1l*sqrtDk*np.exp(-sqrtDkt)

    return np.asarray([zdotzdot_kl_1,zdotzdot_kl_2,zdotzdot_kl_3,zdotzdot_kl_4])

  def zn0dot_zn0dot(self,dt,sqrtDk,sqrtDl,c1k,c1l,c2k,c2l):
    zkdot = self.zidot_n0(dt,sqrtDk,c1k,c2k)
    zldot = self.zidot_n0(dt,sqrtDl,c1l,c2l)
    return zkdot*zldot

  def zn0dot_zn0dot_p(self,dt,sqrtDk,sqrtDl,c1k,c1l,c2k,c2l):
    sqrtDkt = sqrtDk*dt
    sqrtDlt = sqrtDl*dt
    sqrtDkl = sqrtDk*sqrtDl

    zdotzdot_kl_1 = c1k*c1l*sqrtDkl*np.exp(sqrtDkt+sqrtDlt)
    zdotzdot_kl_2 = -c1k*c2l*sqrtDkl*np.exp(sqrtDkt-sqrtDlt)
    zdotzdot_kl_3 = -c2k*c1l*sqrtDkl*np.exp(-sqrtDkt+sqrtDlt)
    zdotzdot_kl_4 = c2k*c2l*sqrtDkl*np.exp(-sqrtDkt-sqrtDlt)

    return np.asarray([zdotzdot_kl_1,zdotzdot_kl_2,zdotzdot_kl_3,zdotzdot_kl_4])

  ###### ZDOT-ZDOT INTEGRALS

  def int_z0dot_z0dot(self,dt,dk,dl,c1k,c1l):
    return np.sum(self.int_z0dot_z0dot_p(dt,dk,dl,c1k,c1l))

  def int_z0dot_z0dot_p(self,dt,dk,dl,c1k,c1l):

    intzdotzdot_kl_1 = dk*dl/3*dt**3
    intzdotzdot_kl_2 = -c1l*dk/2*dt**2
    intzdotzdot_kl_3 = -c1k*dl/2*dt**2
    intzdotzdot_kl_4 = c1k*c1l*dt

    return np.asarray([intzdotzdot_kl_1,intzdotzdot_kl_2,
                       intzdotzdot_kl_3,intzdotzdot_kl_4])

  def int_z0dot_zn0dot(self,dt,dk,sqrtDl,c1l,c1k,c2l):
    return np.sum(self.int_z0dot_zn0dot_p(dt,dk,sqrtDl,c1l,c1k,c2l))

  def int_z0dot_zn0dot_p(self,dt,dk,sqrtDl,c1l,c1k,c2l):
    sqrtDlt = sqrtDl*dt

    intzdotzdot_kl_1 = -c1l*dk/sqrtDl*(np.exp(sqrtDlt)*(sqrtDl*dt-1)+1)
    intzdotzdot_kl_2 = c2l*dk/sqrtDl*(np.exp(-sqrtDlt)*(-sqrtDl*dt-1)+1)
    intzdotzdot_kl_3 = c1l*c1k*np.exp(sqrtDlt)
    intzdotzdot_kl_4 = c2l*c1k*np.exp(-sqrtDlt)

    return np.asarray([intzdotzdot_kl_1,intzdotzdot_kl_2,
                       intzdotzdot_kl_3,intzdotzdot_kl_4])

  def int_zn0dot_z0dot(self,dt,dl,sqrtDk,c1k,c1l,c2k):
    return np.sum(self.int_zn0dot_z0dot_p(dt,dl,sqrtDk,c1k,c1l,c2k))

  def int_zn0dot_z0dot_p(self,dt,dl,sqrtDk,c1k,c1l,c2k):
    sqrtDkt = sqrtDk*dt

    intzdotzdot_kl_1 = -c1k*dl/sqrtDk*(np.exp(sqrtDkt)*(sqrtDk*dt-1)+1)
    intzdotzdot_kl_2 = c2k*dl/sqrtDk*(np.exp(-sqrtDkt)*(-sqrtDk*dt-1)+1)
    intzdotzdot_kl_3 = c1k*c1l*np.exp(sqrtDkt)
    intzdotzdot_kl_4 = c2k*c1l*np.exp(-sqrtDkt)

    return np.asarray([intzdotzdot_kl_1,intzdotzdot_kl_2,
                       intzdotzdot_kl_3,intzdotzdot_kl_4])

  def int_zn0dot_zn0dot(self,dt,sqrtDk,sqrtDl,c1k,c1l,c2k,c2l):
    return np.sum(self.int_zn0dot_zn0dot_p(dt,sqrtDk,sqrtDl,c1k,c1l,c2k,c2l))

  def int_zn0dot_zn0dot_p(self,dt,sqrtDk,sqrtDl,c1k,c1l,c2k,c2l):
    sqrtDkt = sqrtDk*dt
    sqrtDlt = sqrtDl*dt
    sqrtDkl = sqrtDk*sqrtDl

    intzdotzdot_kl_1 = c1k*c1l*sqrtDkl/(sqrtDk+sqrtDl)*(np.exp(sqrtDkt+sqrtDlt)-1)
    if np.allclose(sqrtDk,sqrtDl):
      intzdotzdot_kl_2 = -c1k*c2l*sqrtDkl*dt
      intzdotzdot_kl_3 = -c2k*c1l*sqrtDkl*dt
    else:
      intzdotzdot_kl_2 = -c1k*c2l*sqrtDkl/(sqrtDk-sqrtDl)*(np.exp(sqrtDkt-sqrtDlt)-1)
      intzdotzdot_kl_3 = -c2k*c1l*sqrtDkl/(sqrtDl-sqrtDk)*(np.exp(-sqrtDkt+sqrtDlt)-1)
    intzdotzdot_kl_4 = -c2k*c2l*sqrtDkl/(sqrtDk+sqrtDl)*(np.exp(-sqrtDkt-sqrtDlt)-1)

    return np.asarray([intzdotzdot_kl_1,intzdotzdot_kl_2,
                       intzdotzdot_kl_3,intzdotzdot_kl_4])

  ###### Z-ZDOT DOUBLE INTEGRALS

  def intint_z0_z0dot(self,dt,dk,dl,c1k,c1l,c2k,c2l):
    return np.sum(self.intint_z0_z0dot_p(dt,dk,dl,c1k,c1l,c2k,c2l))

  def intint_z0_z0dot_p(self,dt,dk,dl,c1k,c1l,c2k,c2l):

    intintzzdot_kl_1 = dk*dl/40*dt**5
    intintzzdot_kl_2 = -c1l*dk/24*dt**4
    intintzzdot_kl_3 = -c1k*dl/12*dt**4
    intintzzdot_kl_4 = c1k*c1l/6*dt**3
    intintzzdot_kl_5 = -c2k*dl/6*dt**3
    intintzzdot_kl_6 = c2k*c1l/2*dt**2

    return np.asarray([intintzzdot_kl_1,intintzzdot_kl_2,intintzzdot_kl_3,
                       intintzzdot_kl_4,intintzzdot_kl_5,intintzzdot_kl_6])

  def intint_z0_zn0dot(self,dt,dk,Dl,sqrtDl,c1k,c1l,c2k,c2l):
    return np.sum(self.intint_z0_zn0dot_p(dt,dk,Dl,sqrtDl,c1k,c1l,c2k,c2l))

  def intint_z0_zn0dot_p(self,dt,dk,Dl,sqrtDl,c1k,c1l,c2k,c2l):
    sqrtDlt = sqrtDl*dt

    intintzzdot_kl_1 = -dk*c1l/(2*Dl)*(1/sqrtDl*(np.exp(sqrtDlt)*((sqrtDlt-2)**2+2)-6)-2*dt)
    intintzzdot_kl_2 = dk*c2l/(2*Dl)*(-1/sqrtDl*(np.exp(-sqrtDlt)*((-sqrtDlt-2)**2+2)-6)-2*dt)
    intintzzdot_kl_3 = c1k*c1l/sqrtDl*(1/sqrtDl*(np.exp(sqrtDlt)*(sqrtDlt-2)+2)+dt)
    intintzzdot_kl_4 = -c1k*c2l/sqrtDl*(-1/sqrtDl*(np.exp(-sqrtDlt)*(-sqrtDlt-2)+2)+dt)
    intintzzdot_kl_5 = c2k*c1l*(1/sqrtDl*np.exp(sqrtDlt)-dt)
    intintzzdot_kl_6 = c2k*c2l*(-1/sqrtDl*np.exp(-sqrtDlt)-dt)

    return np.asarray([intintzzdot_kl_1,intintzzdot_kl_2,intintzzdot_kl_3,
                       intintzzdot_kl_4,intintzzdot_kl_5,intintzzdot_kl_6])

  def intint_zn0_z0dot(self,dt,dk,dl,Dk,sqrtDk,c1k,c1l,c2k):
    return np.sum(self.intint_zn0_z0dot_p(dt,dk,dl,Dk,sqrtDk,c1k,c1l,c2k))

  def intint_zn0_z0dot_p(self,dt,dk,dl,Dk,sqrtDk,c1k,c1l,c2k):
    sqrtDkt = sqrtDk*dt

    intintzzdot_kl_1 = -dk*dl/(6*Dk)*dt**3
    intintzzdot_kl_2 = dk*c1l/(2*Dk)*dt**2
    intintzzdot_kl_3 = -c1k*dl/Dk*(1/sqrtDk*(np.exp(sqrtDkt)*(sqrtDkt-2)+2)+dt)
    intintzzdot_kl_4 = c1k*c1l/sqrtDk*(1/sqrtDk*(np.exp(sqrtDkt)-1)-dt)
    intintzzdot_kl_5 = -c2k*dl/Dk*(-1/sqrtDk*(np.exp(-sqrtDkt)*(-sqrtDkt-2)+2)+dt)
    intintzzdot_kl_6 = -c2k*c1l/sqrtDk*(-1/sqrtDk*(np.exp(-sqrtDkt)-1)-dt)

    return np.asarray([intintzzdot_kl_1,intintzzdot_kl_2,intintzzdot_kl_3,
                       intintzzdot_kl_4,intintzzdot_kl_5,intintzzdot_kl_6])

  def intint_zn0_zn0dot(self,dt,dk,Dk,sqrtDk,sqrtDl,c1k,c1l,c2k,c2l):
    return np.sum(self.intint_zn0_zn0dot_p(dt,dk,Dk,sqrtDk,sqrtDl,c1k,c1l,c2k,c2l))

  def intint_zn0_zn0dot_p(self,dt,dk,Dk,sqrtDk,sqrtDl,c1k,c1l,c2k,c2l):
    sqrtDkt = sqrtDk*dt
    sqrtDlt = sqrtDl*dt

    intintzzdot_kl_1 = dk*c1l/Dk*(1/sqrtDl*(np.exp(sqrtDlt)-1)-dt)
    intintzzdot_kl_2 = dk*c2l/Dk*(-1/sqrtDl*(np.exp(-sqrtDlt)-1)-dt)
    intintzzdot_kl_3 = c1k*c1l*sqrtDl/(sqrtDk+sqrtDl)*(1/(sqrtDk+sqrtDl)*(np.exp(sqrtDkt+sqrtDlt)-1)-dt)
    if np.allclose(sqrtDk,sqrtDl):
      intintzzdot_kl_4 = -c1k*c2l*sqrtDl/2*dt**2
      intintzzdot_kl_5 = c2k*c1l*sqrtDl/2*dt**2
    else:
      intintzzdot_kl_4 = -c1k*c2l*sqrtDl/(sqrtDk-sqrtDl)*(1/(sqrtDk-sqrtDl)*(np.exp(sqrtDkt-sqrtDlt)-1)-dt)
      intintzzdot_kl_5 = c2k*c1l*sqrtDl/(-sqrtDk+sqrtDl)*(1/(-sqrtDk+sqrtDl)*(np.exp(-sqrtDkt+sqrtDlt)-1)-dt)
    intintzzdot_kl_6 = c2k*c2l*sqrtDl/(sqrtDk+sqrtDl)*(1/(-sqrtDk-sqrtDl)*(np.exp(-sqrtDkt-sqrtDlt)-1)-dt)

    return np.asarray([intintzzdot_kl_1,intintzzdot_kl_2,intintzzdot_kl_3,
                       intintzzdot_kl_4,intintzzdot_kl_5,intintzzdot_kl_6])

  ###### DERIVATIVES (ANY Dii)

  def dCij_dPuv(self,i,j,u,v,Pinv,P,D):
    return Pinv[i,u]*(D[u]*(v==j) - np.sum(Pinv[v,:]*D*P[:,j]))

  def dCij_dDu(self,i,j,u,Pinv,P):
    return Pinv[i,u]*P[u,j]

  def ddi_dBuv(self,i,u,v,P,x0):
    return -P[i,v]*x0[u]

  def ddi_dbu(self,i,u,P):
    return -P[i,u]

  def ddi_dPuv(self,i,u,v,A,b,Pinv,P,D,x0):
    dC_dPuv = np.zeros((self.XDim,self.XDim),dtype=np.complex)
    for k in range(self.XDim):
      for j in range(self.XDim):
        dC_dPuv[k,j] = self.dCij_dPuv(k,j,u,v,Pinv,P,D)
    return -((u==i)*(np.sum(A[:,v]*x0)+b[v]) + np.sum(np.sum(P[i,:]*dC_dPuv,axis=1)*x0))

  def ddi_dDu(self,i,u,P,Pinv,x0):
    return -np.sum(Pinv[i,:]*P[u,:])*np.sum(Pinv[:,u]*x0)

  ###### PATH DERIVATIVES (|Dii| < eps)

  def dc01i_dbu(self,i,u,dt0f,P):
    return dt0f/2*self.ddi_dbu(i,u,P)

  def dc02i_dbu(self,i,u):
    return 0

  def dc01i_dBuv(self,i,u,v,dt0f,P,x0):
    return dt0f/2*self.ddi_dBuv(i,u,v,P,x0)

  def dc02i_dBuv(self,i,u,v):
    return 0

  def dc01i_dPuv(self,i,u,v,dt0f,xf,x0,A,b,Pinv,P,D):
    return (i==u)*(xf[v]-x0[v])/dt0f + dt0f/2*self.ddi_dPuv(i,u,v,A,b,Pinv,P,D,x0)

  def dc02i_dPuv(self,i,u,v,x0):
    return (i==u)*x0[v]

  def dc01i_dDu(self,i,u,dt0f,P,Pinv,x0):
    return dt0f/2*self.ddi_dDu(i,u,P,Pinv,x0)

  def dc02i_dDu(self,i,u):
    return 0

  ###### PATH DERIVATIVES (|Dii| >= eps)

  def dcn01i_dbu(self,i,u,dt0f,P,D):
    return -1/D[i]*self.ddi_dbu(i,u,P) - self.dcn02i_dbu(i,u,dt0f,P,D)

  def dcn02i_dbu(self,i,u,dt0f,P,D):
    sqrtDit = np.sqrt(D[i])*dt0f
    return (np.exp(sqrtDit)-1)/(D[i]*(np.exp(-sqrtDit)-np.exp(sqrtDit)))*self.ddi_dbu(i,u,P)

  def dcn01i_dBuv(self,i,u,v,dt0f,P,D,x0):
    return -1/D[i]*self.ddi_dBuv(i,u,v,P,x0) - self.dcn02i_dBuv(i,u,v,dt0f,P,D,x0)

  def dcn02i_dBuv(self,i,u,v,dt0f,P,D,x0):
    sqrtDit = np.sqrt(D[i])*dt0f
    return (np.exp(sqrtDit)-1)/(D[i]*(np.exp(-sqrtDit)-np.exp(sqrtDit)))*self.ddi_dBuv(i,u,v,P,x0)

  def dcn01i_dPuv(self,i,u,v,dt0f,x0,xf,A,b,Pinv,P,D):
    return (i==u)*x0[v] - 1/D[i]*self.ddi_dPuv(i,u,v,A,b,Pinv,P,D,x0) - self.dcn02i_dPuv(i,u,v,dt0f,x0,xf,A,b,Pinv,P,D)

  def dcn02i_dPuv(self,i,u,v,dt0f,x0,xf,A,b,Pinv,P,D):
    sqrtDit = np.sqrt(D[i])*dt0f
    numer1 = (i==u)*(xf[v]-np.exp(sqrtDit)*x0[v])
    numer2 = 1/D[i]*(np.exp(sqrtDit)-1)*self.ddi_dPuv(i,u,v,A,b,Pinv,P,D,x0)
    denom = np.exp(-sqrtDit)-np.exp(sqrtDit)
    return (numer1+numer2)/denom

  def dcn01i_dDu(self,i,u,dt0f,x0,xf,A,b,Pinv,P,D,d):
    return -1/D[i]*self.ddi_dDu(i,u,P,Pinv,x0) + (i==u)*d[i]/D[i]**2 - self.dcn02i_dDu(i,u,dt0f,x0,xf,A,b,Pinv,P,D,d)

  def dcn02i_dDu(self,i,u,dt0f,x0,xf,A,b,Pinv,P,D,d):
    sqrtDi = np.sqrt(D[i])
    sqrtDit = sqrtDi*dt0f
    Px0i = np.sum(P[i,:]*x0)
    Pxfi = np.sum(P[i,:]*xf)
    numer1 = (np.exp(sqrtDit)-1)*(1/D[i]*self.ddi_dDu(i,u,P,Pinv,x0)-(i==u)*d[i]/D[i]**2)
    numer2 = (i==u)*dt0f/(2*sqrtDi)*np.exp(sqrtDit)*(d[i]/D[i]-Px0i)
    denom1 = np.exp(-sqrtDit)-np.exp(sqrtDit)
    numer3 = (i==u)*self.c2i_n0(dt0f,Px0i,Pxfi,D[i],sqrtDi,d[i])*dt0f*(np.exp(-sqrtDit)+np.exp(sqrtDit))
    denom2 = 2*sqrtDi*(np.exp(-sqrtDit)-np.exp(sqrtDit))
    return (numer1+numer2)/denom1 + numer3/denom2

  ###### Z-ZDOT INTEGRAL DERIVATIVES (|Dii| < eps)

  def dint_z0_z0dot_dbu(self,k,l,u,dt,dk,dl,c1k,c1l,c2k,dk_dbu,dl_dbu,dc1k_dbu,dc1l_dbu,dc2k_dbu):
    return np.sum(self.dint_z0_z0dot_dbu_p(k,l,u,dt,dk,dl,c1k,c1l,c2k,dk_dbu,dl_dbu,dc1k_dbu,dc1l_dbu,dc2k_dbu))

  def dint_z0_z0dot_dbu_p(self,k,l,u,dt,dk,dl,c1k,c1l,c2k,dk_dbu,dl_dbu,dc1k_dbu,dc1l_dbu,dc2k_dbu):
    dintzzdot_kl_1 = (dl*dk_dbu+dk*dl_dbu)/8*dt**4
    dintzzdot_kl_2 = -(dk*dc1l_dbu+c1l*dk_dbu)/6*dt**3
    dintzzdot_kl_3 = -(dl*dc1k_dbu+c1k*dl_dbu)/3*dt**3
    dintzzdot_kl_4 = (c1l*dc1k_dbu+c1k*dc1l_dbu)/2*dt**2
    dintzzdot_kl_5 = -(dl*dc2k_dbu+c2k*dl_dbu)/2*dt**2
    dintzzdot_kl_6 = (c1l*dc2k_dbu+c2k*dc1l_dbu)*dt

    return np.asarray([dintzzdot_kl_1,dintzzdot_kl_2,dintzzdot_kl_3,
                       dintzzdot_kl_4,dintzzdot_kl_5,dintzzdot_kl_6])

  def dint_z0_z0dot_dBuv(self,k,l,u,v,dt,dk,dl,c1k,c1l,c2k,dk_dBuv,dl_dBuv,dc1k_dBuv,dc1l_dBuv,dc2k_dBuv):
    return np.sum(self.dint_z0_z0dot_dBuv_p(k,l,u,v,dt,dk,dl,c1k,c1l,c2k,dk_dBuv,dl_dBuv,dc1k_dBuv,dc1l_dBuv,dc2k_dBuv))

  def dint_z0_z0dot_dBuv_p(self,k,l,u,v,dt,dk,dl,c1k,c1l,c2k,dk_dBuv,dl_dBuv,dc1k_dBuv,dc1l_dBuv,dc2k_dBuv):
    dintzzdot_kl_1 = (dl*dk_dBuv+dk*dl_dBuv)/8*dt**4
    dintzzdot_kl_2 = -(dk*dc1l_dBuv+c1l*dk_dBuv)/6*dt**3
    dintzzdot_kl_3 = -(dl*dc1k_dBuv+c1k*dl_dBuv)/3*dt**3
    dintzzdot_kl_4 = (c1l*dc1k_dBuv+c1k*dc1l_dBuv)/2*dt**2
    dintzzdot_kl_5 = -(dl*dc2k_dBuv+c2k*dl_dBuv)/2*dt**2
    dintzzdot_kl_6 = (c1l*dc2k_dBuv+c2k*dc1l_dBuv)*dt

    return np.asarray([dintzzdot_kl_1,dintzzdot_kl_2,dintzzdot_kl_3,
                       dintzzdot_kl_4,dintzzdot_kl_5,dintzzdot_kl_6])

  def dint_z0_z0dot_dPuv(self,k,l,u,v,dt,dk,dl,c1k,c1l,c2k,dk_dPuv,dl_dPuv,dc1k_dPuv,dc1l_dPuv,dc2k_dPuv):
    return np.sum(self.dint_z0_z0dot_dPuv_p(k,l,u,v,dt,dk,dl,c1k,c1l,c2k,dk_dPuv,dl_dPuv,dc1k_dPuv,dc1l_dPuv,dc2k_dPuv))

  def dint_z0_z0dot_dPuv_p(self,k,l,u,v,dt,dk,dl,c1k,c1l,c2k,dk_dPuv,dl_dPuv,dc1k_dPuv,dc1l_dPuv,dc2k_dPuv):
    dintzzdot_kl_1 = (dl*dk_dPuv+dk*dl_dPuv)/8*dt**4
    dintzzdot_kl_2 = -(dk*dc1l_dPuv+c1l*dk_dPuv)/6*dt**3
    dintzzdot_kl_3 = -(dl*dc1k_dPuv+c1k*dl_dPuv)/3*dt**3
    dintzzdot_kl_4 = (c1l*dc1k_dPuv+c1k*dc1l_dPuv)/2*dt**2
    dintzzdot_kl_5 = -(dl*dc2k_dPuv+c2k*dl_dPuv)/2*dt**2
    dintzzdot_kl_6 = (c1l*dc2k_dPuv+c2k*dc1l_dPuv)*dt

    return np.asarray([dintzzdot_kl_1,dintzzdot_kl_2,dintzzdot_kl_3,
                       dintzzdot_kl_4,dintzzdot_kl_5,dintzzdot_kl_6])

  def dint_z0_z0dot_dDu(self,k,l,u,dt,dk,dl,c1k,c1l,c2k,dk_dDu,dl_dDu,dc1k_dDu,dc1l_dDu,dc2k_dDu):
    return np.sum(self.dint_z0_z0dot_dDu_p(k,l,u,dt,dk,dl,c1k,c1l,c2k,dk_dDu,dl_dDu,dc1k_dDu,dc1l_dDu,dc2k_dDu))

  def dint_z0_z0dot_dDu_p(self,k,l,u,dt,dk,dl,c1k,c1l,c2k,dk_dDu,dl_dDu,dc1k_dDu,dc1l_dDu,dc2k_dDu):
    dintzzdot_kl_1 = (dl*dk_dDu+dk*dl_dDu)/8*dt**4
    dintzzdot_kl_2 = -(dk*dc1l_dDu+c1l*dk_dDu)/6*dt**3
    dintzzdot_kl_3 = -(dl*dc1k_dDu+c1k*dl_dDu)/3*dt**3
    dintzzdot_kl_4 = (c1l*dc1k_dDu+c1k*dc1l_dDu)/2*dt**2
    dintzzdot_kl_5 = -(dl*dc2k_dDu+c2k*dl_dDu)/2*dt**2
    dintzzdot_kl_6 = (c1l*dc2k_dDu+c2k*dc1l_dDu)*dt

    return np.asarray([dintzzdot_kl_1,dintzzdot_kl_2,dintzzdot_kl_3,
                       dintzzdot_kl_4,dintzzdot_kl_5,dintzzdot_kl_6])

  ###### Z-ZDOT INTEGRAL DERIVATIVES (|Dkk| < eps, |Dll| >= eps)

  def dint_z0_zn0dot_dbu(self,k,l,u,dt,dk,Dl,c1k,c1l,c2k,c2l,dk_dbu,dc1k_dbu,dc1l_dbu,dc2k_dbu,dc2l_dbu):
    return np.sum(self.dint_z0_zn0dot_dbu_p(k,l,u,dt,dk,Dl,c1k,c1l,c2k,c2l,dk_dbu,dc1k_dbu,dc1l_dbu,dc2k_dbu,dc2l_dbu))

  def dint_z0_zn0dot_dbu_p(self,k,l,u,dt,dk,Dl,c1k,c1l,c2k,c2l,dk_dbu,dc1k_dbu,dc1l_dbu,dc2k_dbu,dc2l_dbu):
    sqrtDl = np.sqrt(Dl)
    sqrtDlt = sqrtDl*dt

    dintzzdot_kl_1 = -(dk_dbu*c1l+dc1l_dbu*dk)/(2*Dl)*(np.exp(sqrtDlt)*((sqrtDlt-1)**2+1)-2)
    dintzzdot_kl_2 = -(dk_dbu*c2l+dc2l_dbu*dk)/(2*Dl)*(np.exp(-sqrtDlt)*((-sqrtDlt-1)**2+1)-2)
    dintzzdot_kl_3 = (dc1k_dbu*c1l+c1k*dc1l_dbu)/sqrtDl*(np.exp(sqrtDlt)*(sqrtDlt-1)+1)
    dintzzdot_kl_4 = -(dc1k_dbu*c2l+dc2l_dbu*c1k)/sqrtDl*(np.exp(-sqrtDlt)*(-sqrtDlt-1)+1)
    dintzzdot_kl_5 = (dc2k_dbu*c1l+dc1l_dbu*c2k)*(np.exp(sqrtDlt)-1)
    dintzzdot_kl_6 = (dc2k_dbu*c2l+dc2l_dbu*c2k)*(np.exp(-sqrtDlt)-1)

    return np.asarray([dintzzdot_kl_1,dintzzdot_kl_2,dintzzdot_kl_3,
                       dintzzdot_kl_4,dintzzdot_kl_5,dintzzdot_kl_6])

  def dint_z0_zn0dot_dBuv(self,k,l,u,v,dt,dk,c1k,c1l,c2k,c2l,dk_dBuv,dc1k_dBuv,dc1l_dBuv,dc2k_dBuv,dc2l_dBuv):
    return np.sum(self.dint_z0_zn0dot_dBuv_p(k,l,u,v,dt,dk,c1k,c1l,c2k,c2l,dk_dBuv,dc1k_dBuv,dc1l_dBuv,dc2k_dBuv,dc2l_dBuv))

  def dint_z0_zn0dot_dBuv_p(self,k,l,u,v,dt,dk,c1k,c1l,c2k,c2l,dk_dBuv,dc1k_dBuv,dc1l_dBuv,dc2k_dBuv,dc2l_dBuv):
    sqrtDl = np.sqrt(Dl)
    sqrtDlt = sqrtDl*dt

    dintzzdot_kl_1 = -(dk_dBuv*c1l+dc1l_dBuv*dk)/(2*Dl)*(np.exp(sqrtDlt)*((sqrtDlt-1)**2+1)-2)
    dintzzdot_kl_2 = -(dk_dBuv*c2l+dc2l_dBuv*dk)/(2*Dl)*(np.exp(-sqrtDlt)*((-sqrtDlt-1)**2+1)-2)
    dintzzdot_kl_3 = (dc1k_dBuv*c1l+c1k*dc1l_dBuv)/sqrtDl*(np.exp(sqrtDlt)*(sqrtDlt-1)+1)
    dintzzdot_kl_4 = -(dc1k_dBuv*c2l+dc2l_dBuv*c1k)/sqrtDl*(np.exp(-sqrtDlt)*(-sqrtDlt-1)+1)
    dintzzdot_kl_5 = (dc2k_dBuv*c1l+dc1l_dBuv*c2k)*(np.exp(sqrtDlt)-1)
    dintzzdot_kl_6 = (dc2k_dBuv*c2l+dc2l_dBuv*c2k)*(np.exp(-sqrtDlt)-1)

    return np.asarray([dintzzdot_kl_1,dintzzdot_kl_2,dintzzdot_kl_3,
                       dintzzdot_kl_4,dintzzdot_kl_5,dintzzdot_kl_6])

  def dint_z0_zn0dot_dPuv(self,k,l,u,v,dt,dk,c1k,c1l,c2k,c2l,dk_dPuv,dc1k_dPuv,dc1l_dPuv,dc2k_dPuv,dc2l_dPuv):
    return np.sum(self.dint_z0_zn0dot_dPuv_p(k,l,u,v,dt,dk,c1k,c1l,c2k,c2l,dk_dPuv,dc1k_dPuv,dc1l_dPuv,dc2k_dPuv,dc2l_dPuv))

  def dint_z0_zn0dot_dPuv_p(self,k,l,u,v,dt,dk,c1k,c1l,c2k,c2l,dk_dPuv,dc1k_dPuv,dc1l_dPuv,dc2k_dPuv,dc2l_dPuv):
    sqrtDl = np.sqrt(Dl)
    sqrtDlt = sqrtDl*dt

    dintzzdot_kl_1 = -(dk_dPuv*c1l+dc1l_dPuv*dk)/(2*Dl)*(np.exp(sqrtDlt)*((sqrtDlt-1)**2+1)-2)
    dintzzdot_kl_2 = -(dk_dPuv*c2l+dc2l_dPuv*dk)/(2*Dl)*(np.exp(-sqrtDlt)*((-sqrtDlt-1)**2+1)-2)
    dintzzdot_kl_3 = (dc1k_dPuv*c1l+c1k*dc1l_dPuv)/sqrtDl*(np.exp(sqrtDlt)*(sqrtDlt-1)+1)
    dintzzdot_kl_4 = -(dc1k_dPuv*c2l+dc2l_dPuv*c1k)/sqrtDl*(np.exp(-sqrtDlt)*(-sqrtDlt-1)+1)
    dintzzdot_kl_5 = (dc2k_dPuv*c1l+dc1l_dPuv*c2k)*(np.exp(sqrtDlt)-1)
    dintzzdot_kl_6 = (dc2k_dPuv*c2l+dc2l_dPuv*c2k)*(np.exp(-sqrtDlt)-1)

    return np.asarray([dintzzdot_kl_1,dintzzdot_kl_2,dintzzdot_kl_3,
                       dintzzdot_kl_4,dintzzdot_kl_5,dintzzdot_kl_6])

  def dint_z0_zn0dot_dDu(self,k,l,u,dt,dk,c1k,c1l,c2k,c2l,dk_dDu,dc1k_dDu,dc1l_dDu,dc2k_dDu):
    return np.sum(self.dint_z0_zn0dot_dDu_p(k,l,u,dt,dk,c1k,c1l,c2k,c2l,dk_dDu,dc1k_dDu,dc1l_dDu,dc2k_dDu))

  def dint_z0_zn0dot_dDu_p(self,k,l,u,dt,dk,c1k,c1l,c2k,c2l,dk_dDu,dc1k_dDu,dc1l_dDu,dc2k_dDu):
    sqrtDl = np.sqrt(Dl)
    sqrtDlt = sqrtDl*dt

    dintzzdot_kl_1 = -(dk_dDu*c1l+dc1l_dDu*dk)/(2*Dl)*(np.exp(sqrtDlt)*((sqrtDlt-1)**2+1)-2) - \
      (u==l)*dk*c1l/(2*Dl**2)*(2+np.exp(sqrtDlt)*(1/2*sqrtDlt**3-sqrtDlt**2+2*sqrtDlt-2))
    dintzzdot_kl_2 = -(dk_dDu*c2l+dc2l_dDu*dk)/(2*Dl)*(np.exp(-sqrtDlt)*((-sqrtDlt-1)**2+1)-2) + \
      (u==l)*dk*c2l/(2*Dl**2)*(2+np.exp(-sqrtDlt)*(-1/2*sqrtDlt**3-sqrtDlt**2-2*sqrtDlt-2))
    dintzzdot_kl_3 = (dc1k_dDu*c1l+c1k*dc1l_dDu)/sqrtDl*(np.exp(sqrtDlt)*(sqrtDlt-1)+1) + \
      (u==l)*c1k*c1l/(sqrtDl*Dl)*(np.exp(sqrtDlt)*(1/2*dt**2-1/2*sqrtDlt+1/2)+(1/2*sqrtDlt-1))
    dintzzdot_kl_4 = -(dc1k_dDu*c2l+dc2l_dDu*c1k)/sqrtDl*(np.exp(-sqrtDlt)*(-sqrtDlt-1)+1) - \
      (u==l)*c1k*c2l/(sqrtDl*Dl)*(np.exp(-sqrtDlt)*(1/2*dt**2+1/2*sqrtDlt+1/2)+(-1/2*sqrtDlt-1))
    dintzzdot_kl_5 = (dc2k_dDu*c1l+dc1l_dDu*c2k)*(np.exp(sqrtDlt)-1) + \
      (u==l)*c2k*c1l*sqrtDlt*np.exp(sqrtDlt)
    dintzzdot_kl_6 = (dc2k_dDu*c2l+dc2l_dDu*c2k)*(np.exp(-sqrtDlt)-1) - \
      (u==l)*c2k*c2l*sqrtDlt*np.exp(-sqrtDlt)

    return np.asarray([dintzzdot_kl_1,dintzzdot_kl_2,dintzzdot_kl_3,
                       dintzzdot_kl_4,dintzzdot_kl_5,dintzzdot_kl_6])

  ###### Z-ZDOT INTEGRAL DERIVATIVES (|Dkk| >= eps, |Dll| < eps)

  def dint_zn0_z0dot_dbu(self,k,l,u,dt,dk,dl,Dk,c1k,c1l,c2k,dk_dbu,dl_dbu,dc1k_dbu,dc1l_dbu,dc2k_dbu):
    return np.sum(self.dint_zn0_z0dot_dbu_p(k,l,u,dt,dk,dl,Dk,c1k,c1l,c2k,dk_dbu,dl_dbu,dc1k_dbu,dc1l_dbu,dc2k_dbu))

  def dint_zn0_z0dot_dbu_p(self,k,l,u,dt,dk,dl,Dk,c1k,c1l,c2k,dk_dbu,dl_dbu,dc1k_dbu,dc1l_dbu,dc2k_dbu):
    sqrtDk = np.sqrt(Dk)
    sqrtDkt = sqrtDk*dt

    dintzzdot_kl_1 = -dt**2/(2*Dk)*(dk_dbu*dl+dl_dbu*dk)
    dintzzdot_kl_2 = dt/Dk*(dk_dbu*c1l+dc1l_dbu*dk)
    dintzzdot_kl_3 = -1/Dk*(np.exp(sqrtDkt)*(sqrtDk-1)+1)*(dl_dbu*c1k*+dc1k_dbu*dl)
    dintzzdot_kl_4 = 1/sqrtDk*(np.exp(sqrtDkt)-1)*(dc1k_dbu*c1l+dc1l_dbu*c1k)
    dintzzdot_kl_5 = -1/Dk*(np.exp(-sqrtDkt)*(-sqrtDkt-1)+1)*(dl_dbu*c2k+dc2k_dbu*dl)
    dintzzdot_kl_6 = -1/sqrtDk*(np.exp(-sqrtDkt)-1)*(dc2k_dbu*c1l+dc1l_dbu*c2k)

    return np.asarray([dintzzdot_kl_1,dintzzdot_kl_2,dintzzdot_kl_3,
                       dintzzdot_kl_4,dintzzdot_kl_5,dintzzdot_kl_6])

  def dint_zn0_z0dot_dBuv(self,k,l,u,dt,dk,dl,Dk,c1k,c1l,c2k,dk_dBuv,dl_dBuv,dc1k_dBuv,dc1l_dBuv,dc2k_dBuv):
    return np.sum(self.dint_zn0_z0dot_dBuv_p(k,l,u,dt,dk,dl,Dk,c1k,c1l,c2k,dk_dBuv,dl_dBuv,dc1k_dBuv,dc1l_dBuv,dc2k_dBuv))

  def dint_zn0_z0dot_dBuv_p(self,k,l,u,dt,dk,dl,Dk,c1k,c1l,c2k,dk_dBuv,dl_dBuv,dc1k_dBuv,dc1l_dBuv,dc2k_dBuv):
    sqrtDk = np.sqrt(Dk)
    sqrtDkt = sqrtDk*dt

    dintzzdot_kl_1 = -dt**2/(2*Dk)*(dk_dBuv*dl+dl_dBuv*dk)
    dintzzdot_kl_2 = dt/Dk*(dk_dBuv*c1l+dc1l_dBuv*dk)
    dintzzdot_kl_3 = -1/Dk*(np.exp(sqrtDkt)*(sqrtDk-1)+1)*(dl_dBuv*c1k*+dc1k_dBuv*dl)
    dintzzdot_kl_4 = 1/sqrtDk*(np.exp(sqrtDkt)-1)*(dc1k_dBuv*c1l+dc1l_dBuv*c1k)
    dintzzdot_kl_5 = -1/Dk*(np.exp(-sqrtDkt)*(-sqrtDkt-1)+1)*(dl_dBuv*c2k+dc2k_dBuv*dl)
    dintzzdot_kl_6 = -1/sqrtDk*(np.exp(-sqrtDkt)-1)*(dc2k_dBuv*c1l+dc1l_dBuv*c2k)

    return np.asarray([dintzzdot_kl_1,dintzzdot_kl_2,dintzzdot_kl_3,
                       dintzzdot_kl_4,dintzzdot_kl_5,dintzzdot_kl_6])

  def dint_zn0_z0dot_dPuv(self,k,l,u,dt,dk,dl,Dk,c1k,c1l,c2k,dk_dPuv,dl_dPuv,dc1k_dPuv,dc1l_dPuv,dc2k_dPuv):
    return np.sum(self.dint_zn0_z0dot_dPuv_p(k,l,u,dt,dk,dl,Dk,c1k,c1l,c2k,dk_dPuv,dl_dPuv,dc1k_dPuv,dc1l_dPuv,dc2k_dPuv))

  def dint_zn0_z0dot_dPuv_p(self,k,l,u,dt,dk,dl,Dk,c1k,c1l,c2k,dk_dPuv,dl_dPuv,dc1k_dPuv,dc1l_dPuv,dc2k_dPuv):
    sqrtDk = np.sqrt(Dk)
    sqrtDkt = sqrtDk*dt

    dintzzdot_kl_1 = -dt**2/(2*Dk)*(dk_dPuv*dl+dl_dPuv*dk)
    dintzzdot_kl_2 = dt/Dk*(dk_dPuv*c1l+dc1l_dPuv*dk)
    dintzzdot_kl_3 = -1/Dk*(np.exp(sqrtDkt)*(sqrtDk-1)+1)*(dl_dPuv*c1k*+dc1k_dPuv*dl)
    dintzzdot_kl_4 = 1/sqrtDk*(np.exp(sqrtDkt)-1)*(dc1k_dPuv*c1l+dc1l_dPuv*c1k)
    dintzzdot_kl_5 = -1/Dk*(np.exp(-sqrtDkt)*(-sqrtDkt-1)+1)*(dl_dPuv*c2k+dc2k_dPuv*dl)
    dintzzdot_kl_6 = -1/sqrtDk*(np.exp(-sqrtDkt)-1)*(dc2k_dPuv*c1l+dc1l_dPuv*c2k)

    return np.asarray([dintzzdot_kl_1,dintzzdot_kl_2,dintzzdot_kl_3,
                       dintzzdot_kl_4,dintzzdot_kl_5,dintzzdot_kl_6])

  def dint_zn0_z0dot_dDu(self,k,l,u,dt,dk,dl,Dk,c1k,c1l,c2k,dk_dDu,dl_dDu,dc1k_dDu,dc1l_dDu,dc2k_dDu):
    return np.sum(self.dint_zn0_z0dot_dDu_p(k,l,u,dt,dk,dl,Dk,c1k,c1l,c2k,dk_dDu,dl_dDu,dc1k_dDu,dc1l_dDu,dc2k_dDu))

  def dint_zn0_z0dot_dDu_p(self,k,l,u,dt,dk,dl,Dk,c1k,c1l,c2k,dk_dDu,dl_dDu,dc1k_dDu,dc1l_dDu,dc2k_dDu):
    sqrtDk = np.sqrt(Dk)
    sqrtDkt = sqrtDk*dt

    dintzzdot_kl_1 = -dt**2/(2*Dk)*(dk_dDu*dl+dl_dDu*dk) + (u==k)*dk*dl/(2*Dk**2)*dt**2
    dintzzdot_kl_2 = dt/Dk*(dk_dDu*c1l+dc1l_dDu*dk) - (u==k)*dk*c1l/(Dk**2)*dt
    dintzzdot_kl_3 = -1/Dk*(np.exp(sqrtDkt)*(sqrtDk-1)+1)*(dl_dDu*c1k*+dc1k_dDu*dl) - \
      (u==k)*c1k*dl/(Dk**2)*(np.exp(sqrtDkt)*(1/2*dt**2-sqrtDkt+1)+(1/2*sqrtDkt-1))
    dintzzdot_kl_4 = 1/sqrtDk*(np.exp(sqrtDkt)-1)*(dc1k_dDu*c1l+dc1l_dDu*c1k) + \
      (u==k)*c1k*c1l/(sqrtDk*Dk)*(np.exp(sqrtDkt)*(1/2*dt-1)+1)
    dintzzdot_kl_5 = -1/Dk*(np.exp(-sqrtDkt)*(-sqrtDkt-1)+1)*(dl_dDu*c2k+dc2k_dDu*dl) - \
      (u==k)*c2k*dl/(Dk**2)*(np.exp(-sqrtDkt)*(1/2*dt**2+sqrtDkt+1)+(-1/2*sqrtDkt-1))
    dintzzdot_kl_6 = -1/sqrtDk*(np.exp(-sqrtDkt)-1)*(dc2k_dDu*c1l+dc1l_dDu*c2k) - \
      (u==k)*c2k*c1l/(sqrtDk*Dk)*(np.exp(-sqrtDkt)*(1/2*dt-1)+1)

    return np.asarray([dintzzdot_kl_1,dintzzdot_kl_2,dintzzdot_kl_3,
                       dintzzdot_kl_4,dintzzdot_kl_5,dintzzdot_kl_6])

  ###### Z-ZDOT INTEGRAL DERIVATIVES (|Dii| >= eps)

  def dint_zn0_zn0dot_dbu(self,k,l,u,dt,dk,dl,Dk,Dl,c1k,c1l,c2k,c2l,dk_dbu,dl_dbu,dc1k_dbu,dc1l_dbu,dc2k_dbu,dc2l_dbu):
    return np.sum(self.dint_zn0_zn0dot_dbu_p(k,l,u,dt,dk,dl,Dk,Dl,c1k,c1l,c2k,c2l,dk_dbu,dl_dbu,dc1k_dbu,dc1l_dbu,dc2k_dbu,dc2l_dbu))

  def dint_zn0_zn0dot_dbu_p(self,k,l,u,dt,dk,dl,Dk,Dl,c1k,c1l,c2k,c2l,dk_dbu,dl_dbu,dc1k_dbu,dc1l_dbu,dc2k_dbu,dc2l_dbu):
    sqrtDk = np.sqrt(Dk)
    sqrtDl = np.sqrt(Dl)
    sqrtDkt = sqrtDk*dt
    sqrtDlt = sqrtDl*dt

    dintzzdot_kl_1 = (dk_dbu*c1l+dc1l_dbu*dk)/Dk*(np.exp(sqrtDlt)-1)
    dintzzdot_kl_2 = (dk_dbu*c2l+dc2l_dbu*dk)/Dk*(np.exp(-sqrtDlt)-1)
    dintzzdot_kl_3 = (dc1k_dbu*c1l+dc1l_dbu*c1k)*sqrtDl/(sqrtDk+sqrtDl)*(np.exp(sqrtDkt+sqrtDlt)-1)
    if np.allclose(sqrtDk,sqrtDl):
      dintzzdot_kl_4 = -(dc1k_dbu*c2l+dc2l_dbu*c1k)*sqrtDl*dt
      dintzzdot_kl_5 = (dc2k_dbu*c1l+dc1l_dbu*c2k)*sqrtDl*dt
    else:
      dintzzdot_kl_4 = -(dc1k_dbu*c2l+dc2l_dbu*c1k)*sqrtDl/(sqrtDk-sqrtDl)*(np.exp(sqrtDkt-sqrtDlt)-1)
      dintzzdot_kl_5 = -(dc2k_dbu*c1l+dc1l_dbu*c2k)*sqrtDl/(sqrtDk-sqrtDl)*(np.exp(-sqrtDkt+sqrtDlt)-1)
    dintzzdot_kl_6 = (dc2k_dbu*c2l+dc2l_dbu*c2k)*sqrtDl/(sqrtDk+sqrtDl)*(np.exp(-sqrtDkt-sqrtDlt)-1)

    return np.asarray([dintzzdot_kl_1,dintzzdot_kl_2,dintzzdot_kl_3,
                       dintzzdot_kl_4,dintzzdot_kl_5,dintzzdot_kl_6])

  def dint_zn0_zn0dot_dBuv(self,k,l,u,v,dt,dk,dl,Dk,Dl,c1k,c1l,c2k,c2l,dk_dBuv,dl_dBuv,dc1k_dBuv,dc1l_dBuv,dc2k_dBuv,dc2l_dBuv):
    return np.sum(self.dint_zn0_zn0dot_dBuv_p(k,l,u,v,dt,dk,dl,Dk,Dl,c1k,c1l,c2k,c2l,dk_dBuv,dl_dBuv,dc1k_dBuv,dc1l_dBuv,dc2k_dBuv,dc2l_dBuv))

  def dint_zn0_zn0dot_dBuv_p(self,k,l,u,v,dt,dk,dl,Dk,Dl,c1k,c1l,c2k,c2l,dk_dBuv,dl_dBuv,dc1k_dBuv,dc1l_dBuv,dc2k_dBuv,dc2l_dBuv):
    sqrtDk = np.sqrt(Dk)
    sqrtDl = np.sqrt(Dl)
    sqrtDkt = sqrtDk*dt
    sqrtDlt = sqrtDl*dt

    dintzzdot_kl_1 = (dk_dBuv*c1l+dc1l_dBuv*dk)/Dk*(np.exp(sqrtDlt)-1)
    dintzzdot_kl_2 = (dk_dBuv*c2l+dc2l_dBuv*dk)/Dk*(np.exp(-sqrtDlt)-1)
    dintzzdot_kl_3 = (dc1k_dBuv*c1l+dc1l_dBuv*c1k)*sqrtDl/(sqrtDk+sqrtDl)*(np.exp(sqrtDkt+sqrtDlt)-1)
    if np.allclose(sqrtDk,sqrtDl):
      dintzzdot_kl_4 = -(dc1k_dBuv*c2l+dc2l_dBuv*c1k)*sqrtDl*dt
      dintzzdot_kl_5 = (dc2k_dBuv*c1l+dc1l_dBuv*c2k)*sqrtDl*dt
    else:
      dintzzdot_kl_4 = -(dc1k_dBuv*c2l+dc2l_dBuv*c1k)*sqrtDl/(sqrtDk-sqrtDl)*(np.exp(sqrtDkt-sqrtDlt)-1)
      dintzzdot_kl_5 = -(dc2k_dBuv*c1l+dc1l_dBuv*c2k)*sqrtDl/(sqrtDk-sqrtDl)*(np.exp(-sqrtDkt+sqrtDlt)-1)
    dintzzdot_kl_6 = (dc2k_dBuv*c2l+dc2l_dBuv*c2k)*sqrtDl/(sqrtDk+sqrtDl)*(np.exp(-sqrtDkt-sqrtDlt)-1)

    return np.asarray([dintzzdot_kl_1,dintzzdot_kl_2,dintzzdot_kl_3,
                       dintzzdot_kl_4,dintzzdot_kl_5,dintzzdot_kl_6])

  def dint_zn0_zn0dot_dPuv(self,k,l,u,v,dt,dk,dl,Dk,Dl,c1k,c1l,c2k,c2l,dk_dPuv,dl_dPuv,dc1k_dPuv,dc1l_dPuv,dc2k_dPuv,dc2l_dPuv):
    return np.sum(self.dint_zn0_zn0dot_dPuv_p(k,l,u,v,dt,dk,dl,Dk,Dl,c1k,c1l,c2k,c2l,dk_dPuv,dl_dPuv,dc1k_dPuv,dc1l_dPuv,dc2k_dPuv,dc2l_dPuv))

  def dint_zn0_zn0dot_dPuv_p(self,k,l,u,v,dt,dk,dl,Dk,Dl,c1k,c1l,c2k,c2l,dk_dPuv,dl_dPuv,dc1k_dPuv,dc1l_dPuv,dc2k_dPuv,dc2l_dPuv):
    sqrtDk = np.sqrt(Dk)
    sqrtDl = np.sqrt(Dl)
    sqrtDkt = sqrtDk*dt
    sqrtDlt = sqrtDl*dt

    dintzzdot_kl_1 = (dk_dPuv*c1l+dc1l_dPuv*dk)/Dk*(np.exp(sqrtDlt)-1)
    dintzzdot_kl_2 = (dk_dPuv*c2l+dc2l_dPuv*dk)/Dk*(np.exp(-sqrtDlt)-1)
    dintzzdot_kl_3 = (dc1k_dPuv*c1l+dc1l_dPuv*c1k)*sqrtDl/(sqrtDk+sqrtDl)*(np.exp(sqrtDkt+sqrtDlt)-1)
    if np.allclose(sqrtDk,sqrtDl):
      dintzzdot_kl_4 = -(dc1k_dPuv*c2l+dc2l_dPuv*c1k)*sqrtDl*dt
      dintzzdot_kl_5 = (dc2k_dPuv*c1l+dc1l_dPuv*c2k)*sqrtDl*dt
    else:
      dintzzdot_kl_4 = -(dc1k_dPuv*c2l+dc2l_dPuv*c1k)*sqrtDl/(sqrtDk-sqrtDl)*(np.exp(sqrtDkt-sqrtDlt)-1)
      dintzzdot_kl_5 = -(dc2k_dPuv*c1l+dc1l_dPuv*c2k)*sqrtDl/(sqrtDk-sqrtDl)*(np.exp(-sqrtDkt+sqrtDlt)-1)
    dintzzdot_kl_6 = (dc2k_dPuv*c2l+dc2l_dPuv*c2k)*sqrtDl/(sqrtDk+sqrtDl)*(np.exp(-sqrtDkt-sqrtDlt)-1)

    return np.asarray([dintzzdot_kl_1,dintzzdot_kl_2,dintzzdot_kl_3,
                       dintzzdot_kl_4,dintzzdot_kl_5,dintzzdot_kl_6])

  def dint_zn0_zn0dot_dDu(self,k,l,u,dt,dk,dl,Dk,Dl,c1k,c1l,c2k,c2l,dk_dDu,dl_dDu,dc1k_dDu,dc1l_dDu,dc2k_dDu,dc2l_dDu):
    return np.sum(self.dint_zn0_zn0dot_dDu_p(k,l,u,dt,dk,dl,Dk,Dl,c1k,c1l,c2k,c2l,dk_dDu,dl_dDu,dc1k_dDu,dc1l_dDu,dc2k_dDu,dc2l_dDu))

  def dint_zn0_zn0dot_dDu_p(self,k,l,u,dt,dk,dl,Dk,Dl,c1k,c1l,c2k,c2l,dk_dDu,dl_dDu,dc1k_dDu,dc1l_dDu,dc2k_dDu,dc2l_dDu):
    sqrtDk = np.sqrt(Dk)
    sqrtDl = np.sqrt(Dl)
    sqrtDkt = sqrtDk*dt
    sqrtDlt = sqrtDl*dt

    dintzzdot_kl_1 = (dk_dDu*c1l+dc1l_dDu*dk-(u==k)*dk*c1l/Dk)/Dk*(np.exp(sqrtDlt)-1) - \
      dk*c1l*dt/(2*Dk*sqrtDl)*np.exp(sqrtDlt)
    dintzzdot_kl_2 = (dk_dDu*c2l+dc2l_dDu*dk-(u==k)*dk*c2l/Dk)/Dk*(np.exp(-sqrtDlt)-1) - \
      dk*c2l*dt/(2*Dk*sqrtDl)*np.exp(-sqrtDlt)
    dintzzdot_kl_3 = (dc1k_dDu*c1l+dc1l_dDu*c1k)*sqrtDl/(sqrtDk+sqrtDl)*(np.exp(sqrtDkt+sqrtDlt)-1) + \
      c1k*c1l/(2*(sqrtDk+sqrtDl)**2)*((u==l)*(sqrtDk/sqrtDl)-(u==k)*(sqrtDl/sqrtDk))*(np.exp(sqrtDkt+sqrtDlt)-1) + \
      c1k*c1l*sqrtDl/(sqrtDk+sqrtDl)*dt/2*((u==k)/sqrtDk+(u==l)/sqrtDl)*np.exp(sqrtDkt+sqrtDlt)
    if np.allclose(sqrtDk,sqrtDl):
      dintzzdot_kl_4 = -(dc1k_dDu*c2l+dc2l_dDu*c1k)*sqrtDl*dt - (u==l)*c1k*c2l*dt/(2*sqrtDl)
      dintzzdot_kl_5 = (dc2k_dDu*c1l+dc1l_dDu*c2k)*sqrtDl*dt + (u==k)*c2k*c1l*dt/(2*sqrtDl)
    else:
      dintzzdot_kl_4 = -(dc1k_dDu*c2l+dc2l_dDu*c1k)*sqrtDl/(sqrtDk-sqrtDl)*(np.exp(sqrtDkt-sqrtDlt)-1) - \
        c1k*c2l/(2*(sqrtDk-sqrtDl)**2)*((u==l)*(sqrtDk/sqrtDl)-(u==k)*(sqrtDl/sqrtDk))*(np.exp(sqrtDkt-sqrtDlt)-1) - \
        c1k*c2l*sqrtDl/(sqrtDk-sqrtDl)*dt/2*((u==k)/sqrtDk-(u==l)/sqrtDl)*np.exp(sqrtDkt-sqrtDlt)
      dintzzdot_kl_5 = -(dc2k_dDu*c1l+dc1l_dDu*c2k)*sqrtDl/(sqrtDk-sqrtDl)*(np.exp(-sqrtDkt+sqrtDlt)-1) - \
        c2k*c1l/(2*(sqrtDk-sqrtDl)**2)*((u==l)*(sqrtDk/sqrtDl)-(u==k)*(sqrtDl/sqrtDk))*(np.exp(-sqrtDkt+sqrtDlt)-1) + \
        c2k*c1l*sqrtDl/(sqrtDk-sqrtDl)*dt/2*((u==k)/sqrtDk-(u==l)/sqrtDl)*np.exp(-sqrtDkt+sqrtDlt)
    dintzzdot_kl_6 = (dc2k_dDu*c2l+dc2l_dDu*c2k)*sqrtDl/(sqrtDk+sqrtDl)*(np.exp(-sqrtDkt-sqrtDlt)-1) + \
      c2k*c2l/(2*(sqrtDk+sqrtDl)**2)*((u==l)*(sqrtDk/sqrtDl)-(u==k)*(sqrtDl/sqrtDk))*(np.exp(-sqrtDkt-sqrtDlt)-1) - \
      c2k*c2l*sqrtDl/(sqrtDk+sqrtDl)*dt/2*((u==k)/sqrtDk+(u==l)/sqrtDl)*np.exp(-sqrtDkt-sqrtDlt)

    return np.asarray([dintzzdot_kl_1,dintzzdot_kl_2,dintzzdot_kl_3,
                       dintzzdot_kl_4,dintzzdot_kl_5,dintzzdot_kl_6],dtype=np.complex)

  def int_dint_zzdot(self,A,B,Pinv,P,D,b,dt,dt0f,zt,xt,x0,xf,sqrtD,Px0,Pxf,d):
    int_zzdot = np.zeros((self.XDim,)*2,dtype=np.complex)
    dint_zzdot_db = np.zeros((self.XDim,)*3,dtype=np.complex)
    dint_zzdot_dD = np.zeros((self.XDim,)*3,dtype=np.complex)
    dint_zzdot_dB = np.zeros((self.XDim,)*4,dtype=np.complex)
    dint_zzdot_dP = np.zeros((self.XDim,)*4,dtype=np.complex)

    for k,l in crossprod(*([range(self.XDim)]*2)):

      Px0k, Px0l = Px0[k], Px0[l]
      Pxfk, Pxfl = Pxf[k], Pxf[l]
      dk, dl = d[k], d[l]
      Dk, Dl = D[k], D[l]
      sqrtDk, sqrtDl = np.sqrt(Dk), np.sqrt(Dl)

      kc = np.abs(Dk) < self.eps
      lc = np.abs(Dl) < self.eps

      if kc:
        # Dk = 0j
        c1k = self.c1i_0(dt0f,Px0k,Pxfk,dk)
        c2k = self.c2i_0(Px0k)
      else:
        c1k = self.c1i_n0(dt0f,Px0k,Pxfk,Dk,sqrtDk,dk)
        c2k = self.c2i_n0(dt0f,Px0k,Pxfk,Dk,sqrtDk,dk)

      if lc:
        # Dl = 0j
        c1l = self.c1i_0(dt0f,Px0l,Pxfl,dl)
        c2l = self.c2i_0(Px0l)
      else:
        c1l = self.c1i_n0(dt0f,Px0l,Pxfl,Dl,sqrtDl,dl)
        c2l = self.c2i_n0(dt0f,Px0l,Pxfl,Dl,sqrtDl,dl)

      if kc and lc:
        int_zzdot[k,l] = self.int_z0_z0dot(dt,dk,dl,c1k,c1l,c2k,c2l)
      elif kc and not lc:
        int_zzdot[k,l] = self.int_z0_zn0dot(dt,dk,Dl,sqrtDl,c1k,c1l,c2k,c2l)
      elif not kc and lc:
        int_zzdot[k,l] = self.int_zn0_z0dot(dt,dk,dl,Dk,sqrtDk,c1k,c1l,c2k)
      elif not kc and not lc:
        int_zzdot[k,l] = self.int_zn0_zn0dot(dt,dk,dl,Dk,Dl,sqrtDk,sqrtDl,c1k,c1l,c2k,c2l)

      for u in range(self.XDim):

        dk_dbu = self.ddi_dbu(k,u,P)
        dl_dbu = self.ddi_dbu(l,u,P)
        dk_dDu = self.ddi_dDu(k,u,P,Pinv,x0)
        dl_dDu = self.ddi_dDu(l,u,P,Pinv,x0)

        if kc:
          dc1k_dbu = self.dc01i_dbu(k,u,dt0f,P)
          dc2k_dbu = self.dc02i_dbu(k,u)
          dc1k_dDu = self.dc01i_dDu(k,u,dt0f,P,Pinv,x0)
          dc2k_dDu = self.dc02i_dDu(k,u)
        else:
          dc1k_dbu = self.dcn01i_dbu(k,u,dt0f,P,D)
          dc2k_dbu = self.dcn02i_dbu(k,u,dt0f,P,D)
          dc1k_dDu = self.dcn01i_dDu(k,u,dt0f,x0,xf,A,b,Pinv,P,D,d)
          dc2k_dDu = self.dcn02i_dDu(k,u,dt0f,x0,xf,A,b,Pinv,P,D,d)

        if lc:
          dc1l_dbu = self.dc01i_dbu(l,u,dt0f,P)
          dc2l_dbu = self.dc02i_dbu(l,u)
          dc1l_dDu = self.dc01i_dDu(l,u,dt0f,P,Pinv,x0)
          dc2l_dDu = self.dc02i_dDu(l,u)
        else:
          dc1l_dbu = self.dcn01i_dbu(l,u,dt0f,P,D)
          dc2l_dbu = self.dcn02i_dbu(l,u,dt0f,P,D)
          dc1l_dDu = self.dcn01i_dDu(l,u,dt0f,x0,xf,A,b,Pinv,P,D,d)
          dc2l_dDu = self.dcn02i_dDu(l,u,dt0f,x0,xf,A,b,Pinv,P,D,d)

        if kc and lc:
          dint_zzdot_db[u,k,l] = self.dint_z0_z0dot_dbu(k,l,u,dt,dk,dl,c1k,c1l,c2k,dk_dbu,dl_dbu,dc1k_dbu,dc1l_dbu,dc2k_dbu)
          dint_zzdot_dD[u,k,l] = self.dint_z0_z0dot_dDu(k,l,u,dt,dk,dl,c1k,c1l,c2k,dk_dDu,dl_dDu,dc1k_dDu,dc1l_dDu,dc2k_dDu)
        elif kc and not lc:
          dint_zzdot_db[u,k,l] = self.dint_z0_zn0dot_dbu(k,l,u,dt,dk,Dl,c1k,c1l,c2k,c2l,dk_dbu,dc1k_dbu,dc1l_dbu,dc2k_dbu,dc2l_dbu)
          dint_zzdot_dD[u,k,l] = self.dint_z0_zn0dot_dDu(k,l,u,dt,dk,c1k,c1l,c2k,c2l,dk_dDu,dc1k_dDu,dc1l_dDu,dc2k_dDu)
        elif not kc and lc:
          dint_zzdot_db[u,k,l] = self.dint_zn0_z0dot_dbu(k,l,u,dt,dk,dl,Dk,c1k,c1l,c2k,dk_dbu,dl_dbu,dc1k_dbu,dc1l_dbu,dc2k_dbu)
          dint_zzdot_dD[u,k,l] = self.dint_zn0_z0dot_dDu(k,l,u,dt,dk,dl,Dk,c1k,c1l,c2k,dk_dDu,dl_dDu,dc1k_dDu,dc1l_dDu,dc2k_dDu)
        elif not kc and not lc:
          dint_zzdot_db[u,k,l] = self.dint_zn0_zn0dot_dbu(k,l,u,dt,dk,dl,Dk,Dl,c1k,c1l,c2k,c2l,dk_dbu,dl_dbu,dc1k_dbu,dc1l_dbu,dc2k_dbu,dc2l_dbu)
          dint_zzdot_dD[u,k,l] = self.dint_zn0_zn0dot_dDu(k,l,u,dt,dk,dl,Dk,Dl,c1k,c1l,c2k,c2l,dk_dDu,dl_dDu,dc1k_dDu,dc1l_dDu,dc2k_dDu,dc2l_dDu)

        for v in range(self.XDim):

          dk_dBuv = self.ddi_dBuv(k,u,v,P,x0)
          dl_dBuv = self.ddi_dBuv(l,u,v,P,x0)
          dk_dPuv = self.ddi_dPuv(k,u,v,A,b,Pinv,P,D,x0)
          dl_dPuv = self.ddi_dPuv(l,u,v,A,b,Pinv,P,D,x0)

          if kc:
            dc1k_dBuv = self.dc01i_dBuv(k,u,v,dt0f,P,x0)
            dc2k_dBuv = self.dc02i_dBuv(k,u,v)
            dc1k_dPuv = self.dc01i_dPuv(k,u,v,dt0f,xf,x0,A,b,Pinv,P,D)
            dc2k_dPuv = self.dc02i_dPuv(k,u,v,x0)
          else:
            dc1k_dBuv = self.dcn01i_dBuv(k,u,v,dt0f,P,D,x0)
            dc2k_dBuv = self.dcn02i_dBuv(k,u,v,dt0f,P,D,x0)
            dc1k_dPuv = self.dcn01i_dPuv(k,u,v,dt0f,x0,xf,A,b,Pinv,P,D)
            dc2k_dPuv = self.dcn02i_dPuv(k,u,v,dt0f,x0,xf,A,b,Pinv,P,D)

          if lc:
            dc1l_dBuv = self.dc01i_dBuv(l,u,v,dt0f,P,x0)
            dc2l_dBuv = self.dc02i_dBuv(l,u,v)
            dc1l_dPuv = self.dc01i_dPuv(l,u,v,dt0f,xf,x0,A,b,Pinv,P,D)
            dc2l_dPuv = self.dc02i_dPuv(l,u,v,x0)
          else:
            dc1l_dBuv = self.dcn01i_dBuv(l,u,v,dt0f,P,D,x0)
            dc2l_dBuv = self.dcn02i_dBuv(l,u,v,dt0f,P,D,x0)
            dc1l_dPuv = self.dcn01i_dPuv(l,u,v,dt0f,x0,xf,A,b,Pinv,P,D)
            dc2l_dPuv = self.dcn02i_dPuv(l,u,v,dt0f,x0,xf,A,b,Pinv,P,D)

          if kc and lc:
            dint_zzdot_dB[u,v,k,l] = self.dint_z0_z0dot_dBuv(k,l,u,v,dt,dk,dl,c1k,c1l,c2k,dk_dBuv,dl_dBuv,dc1k_dBuv,dc1l_dBuv,dc2k_dBuv)
            dint_zzdot_dP[u,v,k,l] = self.dint_z0_z0dot_dPuv(k,l,u,v,dt,dk,dl,c1k,c1l,c2k,dk_dPuv,dl_dPuv,dc1k_dPuv,dc1l_dPuv,dc2k_dPuv)
          elif kc and not lc:
            dint_zzdot_dB[u,v,k,l] = self.dint_z0_zn0dot_dBuv(k,l,u,v,dt,dk,c1k,c1l,c2k,c2l,dk_dBuv,dc1k_dBuv,dc1l_dBuv,dc2k_dBuv,dc2l_dBuv)
            dint_zzdot_dP[u,v,k,l] = self.dint_z0_zn0dot_dPuv(k,l,u,v,dt,dk,c1k,c1l,c2k,c2l,dk_dPuv,dc1k_dPuv,dc1l_dPuv,dc2k_dPuv,dc2l_dPuv)
          elif not kc and lc:
            dint_zzdot_dB[u,v,k,l] = self.dint_zn0_z0dot_dBuv(k,l,u,dt,dk,dl,Dk,c1k,c1l,c2k,dk_dBuv,dl_dBuv,dc1k_dBuv,dc1l_dBuv,dc2k_dBuv)
            dint_zzdot_dP[u,v,k,l] = self.dint_zn0_z0dot_dPuv(k,l,u,dt,dk,dl,Dk,c1k,c1l,c2k,dk_dPuv,dl_dPuv,dc1k_dPuv,dc1l_dPuv,dc2k_dPuv)
          elif not kc and not lc:
            dint_zzdot_dB[u,v,k,l] = self.dint_zn0_zn0dot_dBuv(k,l,u,v,dt,dk,dl,Dk,Dl,c1k,c1l,c2k,c2l,dk_dBuv,dl_dBuv,dc1k_dBuv,dc1l_dBuv,dc2k_dBuv,dc2l_dBuv)
            dint_zzdot_dP[u,v,k,l] = self.dint_zn0_zn0dot_dPuv(k,l,u,v,dt,dk,dl,Dk,Dl,c1k,c1l,c2k,c2l,dk_dPuv,dl_dPuv,dc1k_dPuv,dc1l_dPuv,dc2k_dPuv,dc2l_dPuv)

    return int_zzdot, dint_zzdot_db, dint_zzdot_dD, dint_zzdot_dB, dint_zzdot_dP



