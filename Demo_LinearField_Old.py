import time
import numpy as np

import matplotlib.pyplot as plt

from VISolver.Domains.LinearField import LinearField, LFProj

from VISolver.Solvers.Euler import Euler
from VISolver.Solvers.HeunEuler import HeunEuler

# from VISolver.Projection import PolytopeProjection
from VISolver.Solver import Solve
from VISolver.Options import (
    DescentOptions, Miscellaneous, Reporting, Termination, Initialization)
from VISolver.Log import PrintSimResults, PrintSimStats

from IPython import embed
def Demo():

    # __LINEAR_FIELD__##############################################

    # Load Dummy Data
    X = np.random.rand(1000,2)*2-1
    y = 0.5*np.sum(X**2,axis=1)

    # Construct Field
    LF = LinearField(X=X,dy=y,eps=1e-8)

    # # Set Method
    # P = LFProj(LF.param_shapes)
    # Method = Euler(Domain=LF,FixStep=True,P=P)
    # # Method = HeunEuler(Domain=LF,Delta0=1e-1,P=P)

    # # Initialize Starting Field
    # # A = np.array([[0,1],[-1,0]])
    # # A = np.eye(LF.XDim)
    # # A = np.random.rand(LF.XDim,LF.XDim)
    # A = np.array([[5,0.],[0.,5]])
    # b = np.zeros(LF.XDim)
    # Start = LF.Ab_to_BPDb(A,b,flat=True).astype(np.complex)
    # print(Start)

    # # Set Options
    # Init = Initialization(Step=-1.0)
    # Term = Termination(MaxIter=100) #,Tols=[(LF.error,1e-10)])
    # Repo = Reporting(Requests=[LF.error, 'Step', 'F Evaluations',
    #                            'Projections', 'Data'])
    # Misc = Miscellaneous()
    # Options = DescentOptions(Init,Term,Repo,Misc)

    # # Print Stats
    # PrintSimStats(LF,Method,Options)

    # # Start Solver
    # tic = time.time()
    # Results = Solve(Start,Method,LF,Options)
    # toc = time.time() - tic

    # # Print Results
    # PrintSimResults(Options,Results,Method,toc)

    # error = np.asarray(Results.PermStorage[LF.error])
    # params = Results.PermStorage['Data'][-1]

    # A,B,C,Pinv,D,P,b = LF.ExtractParams(params)

    # print(A)
    # print(b)
    # print(error[-1])









    # Initialize Field
    # A = 10*np.random.rand(LF.XDim,LF.XDim)
    # A = (A+A.T)/2
    # A = 100*np.array([[0,1],[-1,0]])
    A = 10*np.eye(LF.XDim)
    # b = 10*np.random.rand(LF.XDim)
    b = np.zeros(LF.XDim)
    # b = np.ones(2)

    # Compute Path Integral
    t = 1
    t0 = 0
    tf = 1
    x0 = np.zeros(LF.XDim)
    xf = np.ones(LF.XDim)
    # xf = np.zeros(LF.XDim)
    y0 = 0
    y1 = LF.predict([A,b],t0,x0,y0,tf,xf,t=t/2)
    print('y(xt(t/2))=',y1)
    y1 = LF.predict([A,b],t0,x0,y0,tf,xf,t=t)
    print('y(xf)=',y1)

    xt = LF.x(np.linspace(0.0001,1,50),t0,tf,x0,xf,[A,b])
    plt.plot(x0[0],x0[1],'*')
    plt.plot(xt[:,0],xt[:,1],'o')
    plt.plot(xf[0],xf[1],marker=(5,1,0),markersize=20)
    plt.xlim([-1,2])
    plt.ylim([-1,2])

    L = LF.Lagrangian(t,t0,x0,y0,tf,xf,[A,b])
    print('Lagrangian=',L)

    EL = LF.EulerLagrange(np.linspace(0.0001,1,10),t0,x0,y0,tf,xf,[A,b])
    print('EL==0?: ',np.allclose(EL,0))

    S = LF.Action(tf,t0,x0,y0,tf,xf,[A,b])
    print('Action=',S)

    fdG = LF.findiff([A,b],t0,x0,y0,tf,xf,t=t)

    print(fdG)
    print('\n')
    G = LF.gradient([A,b],t0,x0,y0,tf,xf,t=t)
    print(G)

    X, Y = np.meshgrid(np.arange(-1, 2, .1), np.arange(-1, 2, .1))
    # points = np.asarray(list(zip(X.ravel(),Y.ravel())))
    points = np.vstack((X.ravel(),Y.ravel())).T
    vectors = LF.Field([A,b],points)
    U = vectors[:,0].reshape(X.shape)
    V = vectors[:,1].reshape(Y.shape)
    Q = plt.quiver(X[::3, ::3], Y[::3, ::3], U[::3, ::3], V[::3, ::3],
               pivot='mid', color='r', units='inches')
    qk = plt.quiverkey(Q, 0.5, 0.03, 1, r'$1 \frac{m}{s}$',
                       fontproperties={'weight': 'bold'})
    plt.plot(X[::3, ::3], Y[::3, ::3], 'k.')

    plt.show()

    # # print(xt)
    # embed()


if __name__ == '__main__':
    Demo()
