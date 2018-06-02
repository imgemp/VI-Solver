import numpy as np

from VISolver.Domain import Domain

from IPython import embed


class CBOW(Domain):

  def __init__(self,seq,EDim=10,seq_limit=20,fix_limit=False,batch_size=100):
    self.seq = seq
    self.index = 0

    self.EDim = EDim

    self.seq_limit = seq_limit
    self.fix_limit = fix_limit

    self.batch_size = batch_size

  '''
  Retrieve embedded samples from dataset
  '''
  def seq_embeddings(self,embeddings):
    if self.fix_limit:
      splits = np.cumsum(self.seq_limit*np.ones(self.batch_size))
    else:
      splits = np.cumsum(np.random.randint(3,self.seq_limit+1,size=self.batch_size))

    if self.index + splits[-1] >= len(self.seq):
      self.index = 0
    seq_split = np.split(self.seq[self.index:self.index+splits[-1]],splits[:-1])

    x = []
    y = []
    for seq in seq_split:
      x += [[embeddings[j*self.EDim:(j+1)*self.EDim] for j in seq[:-1]]]
      end = seq[-1]
      y += [embeddings[end*self.EDim:(end+1)*self.EDim]]

    # shift index
    self.index = splits[-1]

    return x, y, seq_split

  '''
  Compute squared error of CBOW prediction given embeddings
  '''
  def Error(self,embeddings):
    x, y, seq_split = self.seq_embeddings(embeddings)
    err = self.error(x,y,seq_split,embeddings)

    return err

  '''
  Compute squared error of field prediction given params and samples
  '''
  def error(self,x,y,seq_split,embeddings):
    # Reshape params for EDim predictions

    error = 0

    for xi,yi,seqi in zip(x,y,seq_split):
      # Compute predictions
      yi_cbow = np.sum(xi,axis=0)

      error += 0.5*np.sum((yi_cbow-yi)**2)/self.batch_size

    return error

  '''
  Compute percent correct given params_embeddings
  '''
  def PercCorrect(self,embeddings):
    x, y, seq_split = self.seq_embeddings(embeddings)
    pc = self.perc_correct(x,seq_split,embeddings)

    return pc

  '''
  Compute squared error of field prediction given params and samples
  '''
  def perc_correct(self,x,seq_split,embeddings):
    # Reshape embeddings for lookup
    embeddings_resh = np.reshape(embeddings,(-1,self.EDim))

    num_correct = 0

    for xi,seqi in zip(x,seq_split):
      # Find word nearest prediction
      yi_cbow = np.sum(xi,axis=0)
      word_field = np.argmin(np.linalg.norm(yi_cbow-embeddings_resh,axis=1))
      word_actual = seqi[-1]

      num_correct += (word_field == word_actual)

    return float(num_correct)/len(seq_split)

  '''
  Compute perplexity where probabilities are proportional to distance from embeddings
  '''
  def Perplexity(self,embeddings):
    x, y, seq_split = self.seq_embeddings(embeddings)
    p = self.perp(x,seq_split,embeddings)

    return p

  '''
  Compute perplexity of field prediction given params and samples
  '''
  def perp(self,x,seq_split,embeddings):
    # Reshape embeddings for lookup
    embeddings_resh = np.reshape(embeddings,(-1,self.EDim))

    H = 0

    for xi,seqi in zip(x,seq_split):
      # Find word nearest prediction
      yi_cbow = np.sum(xi,axis=0)
      dists = np.linalg.norm(yi_cbow-embeddings_resh,axis=1)
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



