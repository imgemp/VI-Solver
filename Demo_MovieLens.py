import time
import numpy as np

from VISolver.Domains.MixtureMean import MixtureMean
from VISolver.Domains.SVDMethod import SVDMethod

# from VISolver.Solvers.HeunEuler import HeunEuler
from VISolver.Solvers.Euler import Euler

from VISolver.Projection import EntropicProjection
from VISolver.Solver import Solve
from VISolver.Options import (
    DescentOptions, Miscellaneous, Reporting, Termination, Initialization)
from VISolver.Log import PrintSimResults, PrintSimStats

from scipy.sparse import coo_matrix
from IPython import embed


def Demo():

    # __MOVIE_LENS:_MATRIX_COMPLETION__#########################################

    folds = 5
    RMSEs = np.empty((5,folds))
    RMSEs.fill(np.NaN)
    for k in xrange(folds):
        # Load Training Data
        path = '/Users/imgemp/Dropbox/690Op/Midterm/data/100/ml-100k/'
        data = np.loadtxt(path+'u'+str(k+1)+'.base',usecols=(0,1,2))
        users = data[:,0] - 1
        movies = data[:,1] - 1
        ratings = data[:,2]
        spdata_train = coo_matrix((ratings,(users,movies)),shape=(943,1682))

        # Load Testing Data
        path = '/Users/imgemp/Dropbox/690Op/Midterm/data/100/ml-100k/'
        data = np.loadtxt(path+'u'+str(k+1)+'.test',usecols=(0,1,2))
        users = data[:,0] - 1
        movies = data[:,1] - 1
        ratings = data[:,2]
        spdata_test = coo_matrix((ratings,(users,movies)),shape=(943,1682))
        mask = (spdata_test != 0).toarray()

        RMSEs[0,k] = score_globalmean(spdata_train,spdata_test,mask)
        RMSEs[1,k] = score_usermean(spdata_train,spdata_test,mask)
        RMSEs[2,k] = score_moviemean(spdata_train,spdata_test,mask)
        RMSEs[3,k] = score_mixturemean(spdata_train,spdata_test,mask)
        tau = 5*np.sqrt(np.prod(spdata_train.shape))
        RMSEs[4,k] = score_svdmethod(spdata_train,spdata_test,mask,tau=tau)

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
    Method = Euler(Domain=Domain,P=EntropicProjection())
    # Method = HeunEuler(Domain=Domain,P=EntropicProjection(),Delta0=1e-5,
    #                    MinStep=-1e-1,MaxStep=-1e-4)

    # Initialize Starting Point
    Start = np.array([.452,.548])

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


def score_svdmethod(train,test,mask,tau=6e3,step=1.9,fixstep=True,iters=200):
    # Define Domain
    Domain = SVDMethod(Data=train,tau=tau)

    # Set Method
    Method = Euler(Domain=Domain,FixStep=fixstep)

    # Initialize Starting Point
    Start = np.zeros(train.shape)

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
    Y = np.asarray(Results.TempStorage['Data'][-1])
    pred = Domain.shrink(Y,Domain.tau)

    return rmse(pred,test,mask)


if __name__ == '__main__':
    Demo()
