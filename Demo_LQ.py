import time
import numpy as np

import matplotlib.pyplot as plt

from VISolver.Domains.LQ import LQ

from VISolver.Solvers.Euler import Euler
from VISolver.Solvers.Extragradient import EG
from VISolver.Solvers.HeunEuler import HeunEuler
from VISolver.Solvers.HeunEuler_PhaseSpace import HeunEuler_PhaseSpace

from VISolver.Projection import BoxProjection
from VISolver.Solver import Solve
from VISolver.Options import (
    DescentOptions, Miscellaneous, Reporting, Termination, Initialization)
from VISolver.Log import PrintSimResults, PrintSimStats
from IPython import embed

from VISolver.Utilities import approx_jacobian

def Demo():

    #__LQ_GAN__##############################################

    # Interpolation between F and RipCurl should probably be nonlinear
    # in terms of L2 norms of matrices if the norms essentially represent the largest eigenvalues

    # what's an example of a pseudomonotone field? stretch vertically linearly a bit? quasimonotone?

    # the extragradient method uses the same step size for the first and second step, as the step size goes
    # to zero, extragradient asymptotes to the projection method
    # modified extragradient methods use different step sizes. if we keep the first step size fixed to some
    # positive value and shrink the second, the dynamics of extragradient remain as desired
    # this is essentially what HE_PhaseSpace is showing
    # HE and HE_PhaseSpace are designed to "simulate" a trajectory - they do not actually change the effective
    # dynamics of the vector field

    # Define Network and Domain
    Domain = LQ(sig=1)

    # Set Method
    lo = [-np.inf,1e-2]
    # Method = Euler(Domain=Domain,FixStep=True,P=BoxProjection(lo=lo))
    # Method = EG(Domain=Domain,FixStep=True,P=BoxProjection(lo=lo))
    # Method = HeunEuler(Domain=Domain,Delta0=1e-4,P=BoxProjection(lo=lo))
    Method = HeunEuler_PhaseSpace(Domain=Domain,Delta0=1e-2,P=BoxProjection(lo=lo))

    # Initialize Starting Point
    # Start = np.random.rand(Domain.Dim)
    scale = 30
    Start = np.array([50.,50.])
    xoff = 0
    yoff = 0
    # no difference between eigenvalues of J at [1.,3.5] and eigenvalues of an outward spiral: a = np.array([[-1,6.92],[-6.92,-1]])
    j = Domain.J(Start)
    print('original evs')
    print(np.linalg.eigvals(j))
    print(np.linalg.eigvals(j+j.T))
    f = Domain.F(Start)
    jsym = j+j.T
    # print(f.dot(jsym.dot(f)))
    tf = Domain.TF(Start)
    print('tf')
    print(tf)
    print(np.linalg.eigvals(tf))
    jrc = Domain.JRipCurl(Start)
    print('jrc')
    print(jrc)
    print(0.5*(jrc+jrc.T))
    print(np.linalg.eigvals(jrc+jrc.T))
    jreg = Domain.JReg(Start)
    print('jreg')
    print(jreg)
    print(0.5*(jreg+jreg.T))
    print(np.linalg.eigvals(jreg+jreg.T))
    jasy = j-j.T
    print('exact')
    print(0.5*np.dot(jasy.T,jasy))

    for gam in np.linspace(0,1,20):
        # print(Domain.JRegEV(Start,gam))
        print(Domain.JRCEV(Start,gam))

    jap = approx_jacobian(Domain.F,Start)
    print(jap)
    print(np.linalg.eigvals(0.5*(jap+jap.T)))

    y = np.array([0,1])
    x = np.array([1,1e-1])
    pre = np.dot(Domain.F(y),x-y)
    post = np.dot(Domain.F(x),x-y)
    print(pre)
    print(post)

    d = 2
    W2 = Domain.sym(np.random.rand(d,d))
    w1 = np.random.rand(d)
    A = np.tril(np.random.rand(d,d))
    A[range(d),range(d)] = np.clip(A[range(d),range(d)],1e-6,np.inf)
    b = np.random.rand(d)
    dmult = np.hstack([W2.flatten(),w1,A.flatten(),b])
    jmult = Domain.Jmult(dmult)
    jskew = (jmult-jmult.T)
    print(np.linalg.matrix_rank(jskew,tol=1e-16))

    W2 = Domain.sym(np.ones((d,d)))
    w1 = np.ones(d)
    A = np.tril(np.ones((d,d)))
    A[range(d),range(d)] = np.clip(A[range(d),range(d)],0,np.inf)
    b = np.ones(d)
    dmult = np.hstack([W2.flatten(),w1,A.flatten(),b])
    jmult = Domain.Jmult(dmult)
    jskew = (jmult-jmult.T)
    print(np.linalg.matrix_rank(jskew))

    W2 = Domain.sym(np.zeros((d,d)))
    w1 = np.zeros(d)
    A = np.tril(np.ones((d,d)))
    A[range(d),range(d)] = np.clip(A[range(d),range(d)],0,np.inf)
    b = np.zeros(d)
    dmult = np.hstack([W2.flatten(),w1,A.flatten(),b])
    jmult = Domain.Jmult(dmult)
    jskew = (jmult-jmult.T)
    print(np.linalg.matrix_rank(jskew))

    np.set_printoptions(linewidth=200)

    s = (d**2+d)//2
    jskewblock = jskew[:s,s+d:s+d+s]

    embed()

    # Set Options
    Init = Initialization(Step=-1e-5)
    Term = Termination(MaxIter=1000,Tols=[(Domain.dist,1e-4)])
    Repo = Reporting(Requests=['Step', 'F Evaluations',
                               'Projections','Data',Domain.dist])
    Misc = Miscellaneous()
    Options = DescentOptions(Init,Term,Repo,Misc)

    # Print Stats
    PrintSimStats(Domain,Method,Options)

    # Start Solver
    tic = time.time()
    LQ_Results = Solve(Start,Method,Domain,Options)
    toc = time.time() - tic

    # Print Results
    PrintSimResults(Options,LQ_Results,Method,toc)

    data = np.array(LQ_Results.PermStorage['Data'])

    X, Y = np.meshgrid(np.arange(-2*scale + xoff, 2*scale + xoff, .2*scale), np.arange(1e-2 + yoff, 4*scale + yoff, .2*scale))
    U = np.zeros_like(X)
    V = np.zeros_like(Y)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            vec = -Domain.F([X[i,j],Y[i,j]])
            U[i,j] = vec[0]
            V[i,j] = vec[1]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    Q = plt.quiver(X[::3, ::3], Y[::3, ::3], U[::3, ::3], V[::3, ::3],
                   pivot='mid', units='inches')
    ax.plot(data[:,0],data[:,1],'-r')
    ax.plot([data[0,0]],[data[0,1]],'k*')
    ax.plot([data[-1,0]],[data[-1,1]],'b*')
    ax.set_xlim([-2*scale + xoff,2*scale + xoff])
    ax.set_ylim([-.1*scale + yoff,4*scale + yoff])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    # plt.show()
    # plt.savefig('original.png')
    # plt.savefig('EGoriginal.png')
    # plt.savefig('RipCurl.png')
    # plt.savefig('RipCurl2.png')
    # plt.savefig('EG.png')
    # plt.savefig('GReg.png')
    plt.savefig('Testing.png')


if __name__ == '__main__':
    Demo()
