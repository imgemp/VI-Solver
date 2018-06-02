import time
import numpy as np

from VISolver.Domains.MixtureMean import MixtureMean
from VISolver.Domains.SVDMethod import SVDMethod
from VISolver.Domains.MatrixFactorization import MatrixFactorization

from VISolver.Solvers.HeunEuler import HeunEuler
from VISolver.Solvers.Euler import Euler

from VISolver.Projection import EntropicProjection, BoxProjection
from VISolver.Solver import Solve
from VISolver.Options import (
    DescentOptions, Miscellaneous, Reporting, Termination, Initialization)
from VISolver.Log import PrintSimResults, PrintSimStats

from scipy.sparse import coo_matrix
from IPython import embed


def Demo(small=True,folds=1):

    # __MOVIE_LENS:_MATRIX_COMPLETION__#########################################

    if small:
        path = '/Users/imgemp/Dropbox/690Op/Midterm/data/100/ml-100k/u'
        ext = '.base'
        dlim = '\t'
        sh = (943,1682)
    else:
        path = '/Users/imgemp/Dropbox/690Op/Midterm/data/1M/ml-1m/r'
        ext = '.train'
        dlim = '::'
        sh = (6040,3952)

    # SVT convergence
    rel_error = []

    RMSEs = np.empty((6,folds))
    RMSEs.fill(np.NaN)
    np.random.seed(0)
    for k in range(folds):
        # Load Training Data
        data = np.loadtxt(path+str(k+1)+ext,usecols=(0,1,2),delimiter=dlim)
        users = data[:,0] - 1
        movies = data[:,1] - 1
        ratings = data[:,2]
        spdata_train = coo_matrix((ratings,(users,movies)),shape=sh)

        # Load Testing Data
        data = np.loadtxt(path+str(k+1)+'.test',usecols=(0,1,2),delimiter=dlim)
        users = data[:,0] - 1
        movies = data[:,1] - 1
        ratings = data[:,2]
        spdata_test = coo_matrix((ratings,(users,movies)),shape=sh)
        mask = (spdata_test != 0).toarray()

        # RMSEs[0,k] = score_globalmean(spdata_train,spdata_test,mask)
        # RMSEs[1,k] = score_usermean(spdata_train,spdata_test,mask)
        # RMSEs[2,k] = score_moviemean(spdata_train,spdata_test,mask)
        # RMSEs[3,k] = score_mixturemean(spdata_train,spdata_test,mask)
        tau = 5*np.sqrt(np.prod(spdata_train.shape))
        RMSEs[4,k], re = score_svdmethod(spdata_train,spdata_test,mask,tau=tau)
        rel_error += [np.asarray(re)]
        # RMSEs[5,k] = score_matrixfac(spdata_train,spdata_test,mask)

    print(RMSEs)
    embed()


def rmse(pred,test,mask):
    sqerr = mask*np.asarray(pred - test)**2.
    rmse = np.sqrt(sqerr.sum()/test.nnz)
    return rmse


def score_globalmean(train,test,mask):
    pred = train.sum()/train.nnz*mask
    return rmse(pred,test,mask)


def score_usermean(train,test,mask):
    globalmean = train.sum()/train.nnz
    usermean = np.asarray(train.sum(axis=1)).squeeze()/train.getnnz(axis=1)
    usermean[np.isnan(usermean)] = globalmean
    pred = (mask.T*usermean).T
    return rmse(pred,test,mask)


def score_moviemean(train,test,mask):
    globalmean = train.sum()/train.nnz
    moviemean = np.asarray(train.sum(axis=0)).squeeze()/train.getnnz(axis=0)
    moviemean[np.isnan(moviemean)] = globalmean
    pred = mask*moviemean
    return rmse(pred,test,mask)


def score_mixturemean(train,test,mask,step=-1e-3,iters=100):
    # Define Domain
    Domain = MixtureMean(Data=train)

    # Set Method
    Method = Euler(Domain=Domain,P=BoxProjection(lo=0.,hi=1.))
    # Method = Euler(Domain=Domain,P=EntropicProjection())
    # Method = HeunEuler(Domain=Domain,P=EntropicProjection(),Delta0=1e-5,
    #                    MinStep=-1e-1,MaxStep=-1e-4)

    # Initialize Starting Point
    # Start = np.array([.452,.548])
    Start = np.array([.452])

    # Set Options
    Init = Initialization(Step=step)
    Term = Termination(MaxIter=iters)
    Repo = Reporting(Requests=['Step', 'F Evaluations'])
    Misc = Miscellaneous()
    Options = DescentOptions(Init,Term,Repo,Misc)

    # Print Stats
    PrintSimStats(Domain,Method,Options)

    # Start Solver
    tic = time.time()
    Results = Solve(Start,Method,Domain,Options)
    toc = time.time() - tic

    # Print Results
    PrintSimResults(Options,Results,Method,toc)

    # Retrieve result
    parameters = np.asarray(Results.TempStorage['Data'][-1])
    pred = Domain.predict(parameters)

    return rmse(pred,test,mask)


def score_svdmethod(train,test,mask,tau=6e3,step=1.9,fixstep=True,iters=250):
    # Define Domain
    Domain = SVDMethod(Data=train,tau=tau)

    # Set Method
    # Method = Euler(Domain=Domain,FixStep=fixstep)
    Method = HeunEuler(Domain=Domain,Delta0=1e2,
                       MinStep=1e0,MaxStep=1e3)

    # Initialize Starting Point
    # globalmean = train.sum()/train.nnz
    # Start = globalmean*np.ones(train.shape)
    Start = np.zeros(train.shape).flatten()

    # Set Options
    Init = Initialization(Step=step)
    Term = Termination(MaxIter=iters,Tols=[(Domain.rel_error,0.2)])
    Repo = Reporting(Requests=[Domain.rel_error,'Step', 'F Evaluations'])
    Misc = Miscellaneous()
    Options = DescentOptions(Init,Term,Repo,Misc)

    # Print Stats
    PrintSimStats(Domain,Method,Options)

    # Start Solver
    tic = time.time()
    Results = Solve(Start,Method,Domain,Options)
    toc = time.time() - tic

    # Print Results
    PrintSimResults(Options,Results,Method,toc)

    # Retrieve result
    Y = np.asarray(Results.TempStorage['Data'][-1]).reshape(train.shape)
    pred = Domain.shrink(Y,Domain.tau)

    return rmse(pred,test,mask), Results.PermStorage[Domain.rel_error]


def score_matrixfac(train,test,mask,step=1e-5,iters=100,k=500):
    # Define Domain
    n,d = train.shape
    sh_P = (n,k)
    sh_Q = (d,k)
    Domain = MatrixFactorization(Data=train,sh_P=sh_P,sh_Q=sh_Q)

    # Set Method
    # Method = Euler(Domain=Domain,FixStep=True)
    Method = HeunEuler(Domain=Domain,Delta0=1e-1,
                       MinStep=1e-7,MaxStep=1e-2)

    # Initialize Starting Point
    globalmean = train.sum()/train.nnz
    scale = np.sqrt(globalmean/k)
    # P = np.random.rand(n,k)
    # Q = np.random.rand(d,k)
    P = scale*np.ones(sh_P)
    Q = scale*np.ones(sh_Q)
    Start = np.hstack((P.flatten(),Q.flatten()))

    # Set Options
    Init = Initialization(Step=step)
    Term = Termination(MaxIter=iters)
    Repo = Reporting(Requests=['Step', 'F Evaluations'])
    Misc = Miscellaneous()
    Options = DescentOptions(Init,Term,Repo,Misc)

    # Print Stats
    PrintSimStats(Domain,Method,Options)

    # Start Solver
    tic = time.time()
    Results = Solve(Start,Method,Domain,Options)
    toc = time.time() - tic

    # Print Results
    PrintSimResults(Options,Results,Method,toc)

    # Retrieve result
    parameters = np.asarray(Results.TempStorage['Data'][-1])
    pred = Domain.predict(parameters)

    return rmse(pred,test,mask)


if __name__ == '__main__':
    Demo(small=False,folds=5)
    # Demo()
