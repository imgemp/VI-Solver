import numpy as np

from VISolver.Projection import Projection
from VISolver.Domain import Domain

from IPython import embed
class PTBProj(Projection):

  def __init__(self,EDim,threshold=0.):
    self.EDim = EDim
    param_shapes = [(EDim,EDim,EDim),(EDim,EDim),(EDim)]
    self.param_len = np.sum([np.prod(sh) for sh in param_shapes])
    self.threshold = threshold

  def P(self,Data,Step=0.,Direc=0.):
    # Take step
    Data += Step*Direc

    # Reshape params to extract As
    params, embeddings = np.split(Data,[self.param_len])

    # # Project embeddings to interior of unit sphere
    # num_embed = embeddings.shape[0]//self.EDim
    # embeddings_split = np.split(embeddings,num_embed)
    # for i in range(num_embed):
    #   norm = np.linalg.norm(embeddings_split[i])
    #   if norm > 1:
    #     embeddings_split[i] /= norm
    # embeddings = np.hstack(embeddings_split)

    # Iterative shrinkage
    if self.threshold > 0.:
      params = np.clip(np.abs(params)-self.threshold,0.,np.inf)*np.sign(params)

    # Reshape params to extract As
    params_resh = np.reshape(params,(self.EDim,-1))  # [[A,b,C],[A,b,C],...]
    As = params_resh[:,:self.EDim**2]

    # Enforce PSD Constraint
    for i,A in enumerate(As):
      Asq = np.reshape(A,(self.EDim,-1))
      Asym = 0.5*(Asq + Asq.T)
      Aasy = 0.5*(Asq - Asq.T)
      if not np.all(np.diag(Asym) >= np.sum(np.abs(Asym),axis=1) - np.abs(np.diag(Asym))):
        try:
          # U, s, V = np.linalg.svd(Asym, full_matrices=True)
          s, V = np.linalg.eig(Asym)
        except Exception:
          print(Exception)
          embed()
          assert False
        if np.all(s>=0.):
          continue
        S = np.diag(np.clip(s,0,np.inf))

        Apsd = V.dot(S).dot(np.linalg.inv(V))

        # Apsd = np.dot(U, np.dot(S, V)) + Aasy
        Apsd_flat = Apsd.flatten()
        params_resh[i][:len(Apsd_flat)] = Apsd_flat

    return np.hstack((params_resh.flatten(),embeddings))


def get_y0(seq,vocab):
  y0 = list(np.zeros(vocab))
  for i in range(len(seq)-1):
    w = seq[i]
    wnext = seq[i+1]
    if not isinstance(y0[w],dict):
      y0[w] = dict()
    if wnext in y0[w]:
      y0[w][wnext] += 1
    else:
      y0[w][wnext] = 1

  for i in range(len(y0)):
    arr = np.asarray(list(y0[i].items()))
    y0[i] = arr[np.argmax(arr[:,1]),0]
  return y0


class PennTreeBank(Domain):

  def __init__(self,seq,y0=None,EDim=10,seq_limit=20,fix_limit=False,batch_size=100,
               gamma=0.,beta=0.,learn_field=True,learn_embedding=True,ord=2):
    self.seq = seq
    self.y0 = y0
    self.index = 0

    self.EDim = EDim
    if y0 is None:
      self.param_shapes = [(EDim,EDim,EDim),(EDim,EDim),(EDim)]
    else:
      self.param_shapes = [(EDim,EDim,EDim),(EDim,EDim)]
    self.param_len = np.sum([np.prod(sh) for sh in self.param_shapes])
    self.Dim = self.param_len + len(np.unique(seq))*self.EDim

    self.seq_limit = seq_limit
    self.fix_limit = fix_limit

    self.batch_size = batch_size

    self.gamma = gamma
    self.beta = beta

    self.learn_field = learn_field
    self.learn_embedding = learn_embedding

    self.ord = ord

  def F(self,params_embeddings):
    params, embeddings = np.split(params_embeddings,[self.param_len])
    x, y, seq_split = self.seq_embeddings(embeddings)
    if self.learn_field:
      grad_param = self.field_grad(params,x,y,seq_split,embeddings)
    else:
      grad_param = np.zeros_like(params)
    if self.learn_embedding:
      grad_embeddings = self.embed_grad(seq_split,x,y,embeddings,params)
    else:
      grad_embeddings = np.zeros_like(embeddings)
    if np.any(np.isnan(params_embeddings)) or np.any(np.isnan(grad_param)) or np.any(np.isnan(grad_embeddings)):
      embed()
      assert False

    return np.hstack((grad_param,grad_embeddings))

  '''
  Retrieve embedded samples from dataset
  '''
  def seq_embeddings(self,embeddings):
    if self.fix_limit:
      splits = np.cumsum(self.seq_limit*np.ones(self.batch_size,dtype=int))
    else:
      splits = np.cumsum(np.random.randint(3,self.seq_limit+1,size=self.batch_size))

    if self.index + splits[-1] >= len(self.seq):
      self.index = 0
      print('\nCompleted pass through dataset.\n')
    seq_split = np.split(self.seq[self.index:self.index+splits[-1]],splits[:-1])

    x = []
    y = []
    for seq in seq_split:
      x += [[embeddings[j*self.EDim:(j+1)*self.EDim] for j in seq[:-1]]]
      end = seq[-1]
      y += [embeddings[end*self.EDim:(end+1)*self.EDim]]

    # shift index
    self.index += splits[-1]

    return x, y, seq_split

  '''
  Compute gradient of field prediction with respect to field parameters
  '''
  def field_grad(self,params,x,y,seq_split,embeddings):
    # Reshape params for EDim predictions
    params_resh = np.reshape(params,(self.EDim,-1))  # [[A,b,C],[A,b,C],...]
    param_dim = params_resh.shape[1]

    grad = np.zeros_like(params)

    for xi,yi,seqi in zip(x,y,seq_split):
      # Construct auxiliary embeddings
      if self.y0 is None:
        zi = np.zeros(param_dim-1)
      else:
        zi = np.zeros(param_dim)
      for x0,xf in zip(xi[:-1],xi[1:]):
        zb = xf - x0
        zA = 0.5*np.outer(zb,xf+x0).flatten()
        zi += np.hstack((zA,zb))
      if self.y0 is None:
        zi = np.append(zi,1)

      # Compute predictions and gradients
      if self.y0 is None:
        yi_field = params_resh.dot(zi)
      else:
        idx = self.y0[seqi[0]]
        yi_field = params_resh.dot(zi) + embeddings[idx*self.EDim:(idx+1)*self.EDim]
      yi_field_grad = np.tile(zi,self.EDim)

      derr = self.derror(yi_field,yi)

      grad += np.repeat(derr,param_dim)*yi_field_grad/self.batch_size

    # Add L2 regularization to field params
    grad += self.gamma*params

    return grad

  '''
  Compute gradient of field prediction with respect to embedding vectors
  '''
  def embed_grad(self,seq_split,x,y,embeddings,params):
    # Reshape params for EDim predictions
    params_resh = np.reshape(params,(self.EDim,-1))  # [[A,b,C],[A,b,C],...]
    As = params_resh[:,:self.EDim**2].reshape((-1,self.EDim,self.EDim))
    bs = params_resh[:,self.EDim**2:self.EDim**2+self.EDim]
    if self.y0 is None:
      Cs = params_resh[:,-1]

    # Init gradient vectors
    num_embed = embeddings.shape[0]//self.EDim
    grad = np.zeros_like(embeddings)
    yi_field = np.zeros(self.EDim)
    yi_field_grad = np.zeros((self.EDim,)+embeddings.shape)

    # Compute predictions and gradients
    for seqi,xi,yi in zip(seq_split,x,y):
      yi_field *= 0
      yi_field_grad *= 0
      F0 = np.einsum('...ij,...j->...i',As,xi[0]) + bs  # compute Ax0 + b
      for j in range(len(seqi)-2):
        seq0, seqf, x0, xf = seqi[j], seqi[j+1], xi[j], xi[j+1]
        Ff = np.einsum('...ij,...j->...i',As,xf) + bs  # compute Axf + b
        Fmid = 0.5*(Ff+F0)
        dx = xf - x0
        yi_field += np.dot(Fmid,dx)
        yi_field_grad[:,seq0*self.EDim:(seq0+1)*self.EDim] += -Fmid
        yi_field_grad[:,seqf*self.EDim:(seqf+1)*self.EDim] += Fmid
        F0 = np.einsum('...ij,...j->...i',As,x0) + bs  # compute Ax0 + b for next iteration

      if self.y0 is None:
        yi_field += Cs
      else:
        idx = self.y0[seqi[0]]
        yi_field += embeddings[idx*self.EDim:(idx+1)*self.EDim]

      # Add grad for yi = -1
      yi_field_grad[:,seqi[-1]*self.EDim:(seqi[-1]+1)*self.EDim] += -1

      # Compute error derivative
      derr = self.derror(yi_field,yi)

      # Increment gradient
      grad += np.einsum('i...,i...->...',derr,yi_field_grad)/self.batch_size

    # Add repelling force to avoid mapping all x to origin
    #   grad of -beta*0.5*log(x1^2 + ... + xn^2) is -beta/(||x||^2)*xi
    # NEED TO AVOID MAPPING ALL TO CENTROID OF EMBEDDINGS -  THIS MAY BE WRONG - ORIGIN APPROACH IS RIGHT
    #   grad of -beta*0.5*log((x1-xc)^2 + ... + (xn-xc)^2) where xc = 1/N*sum(xi)
    words = np.unique(seqi)
    centroid = np.mean(np.split(embeddings,num_embed),axis=0)
    sum_dist = np.linalg.norm(embeddings-np.repeat(centroid,num_embed),ord=self.ord)**(self.ord)
    coeff = -self.beta*self.EDim/sum_dist
    for word in words:
      start, end =  word*self.EDim, (word+1)*self.EDim
      grad[start:end] += coeff*(embeddings[start:end]-centroid)

    return grad

  '''
  Compute squared error of field prediction given params_embeddings
  '''
  def Error(self,params_embeddings):
    params, embeddings = np.split(params_embeddings,[self.param_len])
    x, y, seq_split = self.seq_embeddings(embeddings)
    err = self.error(params,x,y,seq_split,embeddings)

    return err

  '''
  Compute squared error of field prediction given params and samples
  '''
  def error(self,params,x,y,seq_split,embeddings):
    # Reshape params for EDim predictions
    params_resh = np.reshape(params,(self.EDim,-1))  # [[A,b,C],[A,b,C],...]
    param_dim = params_resh.shape[1]

    error = 0

    for xi,yi,seqi in zip(x,y,seq_split):
      # Construct auxiliary embeddings
      if self.y0 is None:
        zi = np.zeros(param_dim-1)
      else:
        zi = np.zeros(param_dim)
      for x0,xf in zip(xi[:-1],xi[1:]):
        zb = xf - x0
        zA = 0.5*np.outer(zb,xf+x0).flatten()
        zi += np.hstack((zA,zb))
      if self.y0 is None:
        zi = np.append(zi,1)

      # Compute predictions
      if self.y0 is None:
        yi_field = params_resh.dot(zi)
      else:
        idx = self.y0[seqi[0]]
        yi_field = params_resh.dot(zi) + embeddings[idx*self.EDim:(idx+1)*self.EDim]

      if self.ord == 2:
        error += 0.5*np.sum((yi_field-yi)**2)/self.batch_size
      else:
        error += np.sum(np.abs(yi_field-yi))/self.batch_size

    return error

  '''
  Compute derivative of error
  '''
  def derror(self,y_field,y):
    if self.ord == 2:
      return y_field-y
    else:
      return np.sign(y_field-y)

  '''
  Compute percent correct given params_embeddings
  '''
  def PercCorrect(self,params_embeddings):
    params, embeddings = np.split(params_embeddings,[self.param_len])
    x, y, seq_split = self.seq_embeddings(embeddings)
    pc = self.perc_correct(params,x,seq_split,embeddings)

    return pc

  '''
  Compute squared error of field prediction given params and samples
  '''
  def perc_correct(self,params,x,seq_split,embeddings):
    # Reshape params for EDim predictions
    params_resh = np.reshape(params,(self.EDim,-1))  # [[A,b,C],[A,b,C],...]
    param_dim = params_resh.shape[1]
    embeddings_resh = np.reshape(embeddings,(-1,self.EDim))

    num_correct = 0

    for xi,seqi in zip(x,seq_split):
      # Construct auxiliary embeddings
      if self.y0 is None:
        zi = np.zeros(param_dim-1)
      else:
        zi = np.zeros(param_dim)
      for x0,xf in zip(xi[:-1],xi[1:]):
        zb = xf - x0
        zA = 0.5*np.outer(zb,xf+x0).flatten()
        zi += np.hstack((zA,zb))
      if self.y0 is None:
        zi = np.append(zi,1)

      # Find word nearest prediction
      if self.y0 is None:
        yi_field = params_resh.dot(zi)
      else:
        idx = self.y0[seqi[0]]
        yi_field = params_resh.dot(zi) + embeddings[idx*self.EDim:(idx+1)*self.EDim]
      word_field = np.argmin(np.linalg.norm(yi_field-embeddings_resh,ord=self.ord,axis=1))
      word_actual = seqi[-1]

      num_correct += (word_field == word_actual)

    return float(num_correct)/len(seq_split)

  '''
  Compute perplexity where probabilities are proportional to distance from embeddings
  '''
  def Perplexity(self,params_embeddings):
    params, embeddings = np.split(params_embeddings,[self.param_len])
    x, y, seq_split = self.seq_embeddings(embeddings)
    p = self.perp(params,x,seq_split,embeddings)

    return p

  '''
  Compute perplexity of field prediction given params and samples
  '''
  def perp(self,params,x,seq_split,embeddings):
    # Reshape params for EDim predictions
    params_resh = np.reshape(params,(self.EDim,-1))  # [[A,b,C],[A,b,C],...]
    param_dim = params_resh.shape[1]
    embeddings_resh = np.reshape(embeddings,(-1,self.EDim))

    H = 0

    for xi,seqi in zip(x,seq_split):
      # Construct auxiliary embeddings
      if self.y0 is None:
        zi = np.zeros(param_dim-1)
      else:
        zi = np.zeros(param_dim)
      for x0,xf in zip(xi[:-1],xi[1:]):
        zb = xf - x0
        zA = 0.5*np.outer(zb,xf+x0).flatten()
        zi += np.hstack((zA,zb))
      if self.y0 is None:
        zi = np.append(zi,1)

      # Find word nearest prediction
      if self.y0 is None:
        yi_field = params_resh.dot(zi)
      else:
        idx = self.y0[seqi[0]]
        yi_field = params_resh.dot(zi) + embeddings[idx*self.EDim:(idx+1)*self.EDim]
      dists = np.linalg.norm(yi_field-embeddings_resh,ord=self.ord,axis=1)
      word_actual = seqi[-1]

      p = self.softmax(dists)
      H += -np.log(p[word_actual])

    return np.exp(H/self.batch_size)

  '''
  Compute softmax given distances: smaller distance --> higher probability
  '''
  def softmax(self,dists):
    min_idx = np.argmin(dists)
    min_dist = dists[min_idx]
    if min_dist == 0.:  # assume embeddings don't overlap
      prob = 1e-12*np.ones_like(dists)
      prob[min_idx] = 1-1e-12*(len(prob)-1)
    else:
      logits = -dists/min_dist
      exp_log = np.clip(np.exp(logits),1e-12,np.exp(-1.))
      prob = exp_log/np.sum(exp_log)
    return prob



