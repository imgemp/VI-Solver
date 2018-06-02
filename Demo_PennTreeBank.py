import time
import numpy as np

import pickle

from VISolver.Domains.PennTreeBank import PennTreeBank, PTBProj, get_y0
from VISolver.Domains.PTB_Reader import ptb_raw_data

from VISolver.Domains.CBOW import CBOW

from VISolver.Solvers.Euler import Euler
from VISolver.Solvers.HeunEuler import HeunEuler
from VISolver.Solvers.CashKarp import CashKarp

from VISolver.Projection import BoxProjection
from VISolver.Solver import Solve
from VISolver.Options import (
    DescentOptions, Miscellaneous, Reporting, Termination, Initialization)
from VISolver.Log import PrintSimResults, PrintSimStats

import matplotlib.pyplot as plt

from scipy.spatial.distance import pdist, squareform

from IPython import embed


def Demo():

    # __PENN_TREE_BANK____#################################################

    # Load Data
    # seq = np.arange(1000)
    train, valid, test, id_to_word, vocab = ptb_raw_data('/Users/imgemp/Desktop/Data/simple-examples/data/')
    words, given_embeddings = pickle.load(open('/Users/imgemp/Desktop/Data/polyglot-en.pkl', 'rb'), encoding='latin1')
    words_low = [word.lower() for word in words]
    word_to_id = dict(zip(words_low,range(len(words))))
    word_to_id['<eos>'] = word_to_id['</s>']
    y0 = get_y0(train,vocab)

    EDim = 5
    fix_embedding = False
    learn_embedding = True
    if fix_embedding:
        EDim = given_embeddings.shape[1]
        learn_embedding = False

    # Define Domain
    Domain = PennTreeBank(seq=train,y0=None,EDim=EDim,batch_size=100,
                          learn_embedding=learn_embedding,ord=1)

    # Set Method
    P = PTBProj(Domain.EDim)
    # Method = Euler(Domain=Domain,FixStep=True,P=P)
    Method = HeunEuler(Domain=Domain,P=P,Delta0=1e-4,MinStep=-3.,MaxStep=0.)
    # Method = CashKarp(Domain=Domain,P=P,Delta0=1e-1,MinStep=-5.,MaxStep=0.)

    # Set Options
    Term = Termination(MaxIter=10000)
    Repo = Reporting(Interval=10,Requests=[Domain.Error,Domain.PercCorrect,Domain.Perplexity,'Step']) #,
                               # 'Step', 'F Evaluations',
                               # 'Projections','Data'])
    Misc = Miscellaneous()
    Init = Initialization(Step=-1e-3)
    Options = DescentOptions(Init,Term,Repo,Misc)

    # Initialize Starting Point
    if fix_embedding:
        missed = 0
        params = np.random.rand(Domain.param_len)
        avg_norm = np.linalg.norm(given_embeddings,axis=1).mean()
        embeddings = []
        for i in range(vocab):
            word = id_to_word[i]
            if word in word_to_id:
                embedding = given_embeddings[word_to_id[word]]
            else:
                missed += 1
                embedding = np.random.rand(Domain.EDim)
                embedding *= avg_norm/np.linalg.norm(embedding)
            embeddings += [embedding]
        polyglot_embeddings = np.hstack(embeddings)
        Start = np.hstack((params,polyglot_embeddings))
        print(np.linalg.norm(polyglot_embeddings))
        print('Missing %d matches in polyglot dictionary -> given random embeddings.' % missed)
    else:
        # params = np.random.rand(Domain.param_len)*10
        # embeddings = np.random.rand(EDim*vocab)*.1
        # Start = np.hstack((params,embeddings))
        # assert Start.shape[0] == Domain.Dim
        Start = np.random.rand(Domain.Dim)
    Start = P.P(Start)

    # Compute Initial Error
    print('Initial training error: %g' % Domain.Error(Start))
    print('Initial perplexity: %g' % Domain.Perplexity(Start))
    print('Initial percent correct: %g' % Domain.PercCorrect(Start))

    # Print Stats
    PrintSimStats(Domain,Method,Options)

    # Start Solver
    tic = time.time()
    PTB_Results = Solve(Start,Method,Domain,Options)
    toc = time.time() - tic

    # Print Results
    PrintSimResults(Options,PTB_Results,Method,toc)

    # Plot Results
    err = np.asarray(PTB_Results.PermStorage[Domain.Error])
    pc = np.asarray(PTB_Results.PermStorage[Domain.PercCorrect])
    perp = np.asarray(PTB_Results.PermStorage[Domain.Perplexity])
    steps = np.asarray(PTB_Results.PermStorage['Step'])
    t = np.arange(0,len(steps)*Repo.Interval,Repo.Interval)
    fig = plt.figure()
    ax = fig.add_subplot(411)
    ax.semilogy(t,err)
    ax.set_ylabel('Training Error')
    ax.set_title('Penn Tree Bank Training Evaluation')
    ax = fig.add_subplot(412)
    ax.semilogy(t,perp)
    ax.set_ylabel('Perplexity')
    ax = fig.add_subplot(413)
    ax.plot(t,pc)
    ax.set_ylabel('Percent Correct')
    ax = fig.add_subplot(414)
    ax.plot(t,steps)
    ax.set_ylabel('Step Size')
    ax.set_xlabel('Iterations (k)')
    plt.savefig('PTB')

    params_embeddings = np.asarray(PTB_Results.TempStorage['Data']).squeeze()
    params, embeddings = np.split(params_embeddings,[Domain.param_len])
    embeddings_split = np.split(embeddings,vocab)

    dists_comp = pdist(np.asarray(embeddings_split))
    dists_min = np.min(dists_comp)
    dists_max = np.max(dists_comp)
    dists = squareform(dists_comp)
    dists2 = np.asarray([np.linalg.norm(e) for e in embeddings_split])
    print('pairwise dists',np.mean(dists),dists_min,dists_max)
    print('embedding norms',np.mean(dists2),np.min(dists2),np.max(dists2))
    print('params',np.mean(params),np.min(np.abs(params)),np.max(np.abs(params)))

    # http://sebastianruder.com/word-embeddings-1/index.html#continuousbagofwordscbow
    # Continuous-Bag-Of-Words (CBOW)
    # Write code to add up vectors for samples (new domain file)
    # Define Domain
    # Domain = CBOW(seq=train,EDim=64,batch_size=1000)
    # print(Domain.Error(polyglot_embeddings))
    # print(Domain.PercCorrect(polyglot_embeddings))
    # print(Domain.Perplexity(polyglot_embeddings))



if __name__ == '__main__':
    Demo()