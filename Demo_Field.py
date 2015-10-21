import time
import numpy as np

from VISolver.Domains.FieldRegressorLight import (
    FieldRegressor, constructRandomDataset, constructSampleDataset)
from VISolver.Domains.PolyRegressorLight import PolyRegressor, conv2field

from VISolver.Solvers.HeunEuler import HeunEuler

from VISolver.Solver import Solve
from VISolver.Options import (
    DescentOptions, Miscellaneous, Reporting, Termination, Initialization)
from VISolver.Log import PrintSimResults, PrintSimStats

import matplotlib.pyplot as plt
from IPython import embed


def Demo():

    #__POLY_REGRESSION__##############################################

    # Construct Dataset
    # N = 4
    # dataset = constructRandomDataset(N,dim=2)
    dataset = constructSampleDataset(conservative=False,ex=1)

    # Define Regressor
    Domain = PolyRegressor(dataset,deg=2)

    # Set Method
    Method = HeunEuler(Domain=Domain,Delta0=1e-2)

    # Initialize Starting Point
    Start = np.ones(Domain.Dim)

    # Calculate Initial Error
    f_0 = Domain.f(Start)

    # Set Options
    Init = Initialization(Step=-1e-2)
    Term = Termination(MaxIter=100,Tols=[(Domain.f,1e-4*f_0)])
    Repo = Reporting(Requests=[Domain.f, 'Step', 'F Evaluations','Data'])
    Misc = Miscellaneous()
    Options = DescentOptions(Init,Term,Repo,Misc)

    # Print Stats
    PrintSimStats(Domain,Method,Options)

    # Start Solver
    tic = time.time()
    Poly_Results = Solve(Start,Method,Domain,Options)
    toc = time.time() - tic

    # Print Results
    PrintSimResults(Options,Poly_Results,Method,toc)
    print(conv2field(Poly_Results.PermStorage['Data'][-1]))

    #__FIELD_REGRESSION__##############################################

    # Define Regressor
    Domain = FieldRegressor(dataset,deg=2)

    # Set Method
    Method = HeunEuler(Domain=Domain,Delta0=1e-4)

    # Initialize Starting Point
    # Start = conv2field(Start)
    # Start = conv2field(Poly_Results.PermStorage['Data'][-1])
    # Start = np.ones(Domain.Dim)
    Start = np.zeros(Domain.Dim)

    # Calculate Initial Error
    f_0 = Domain.f(Start)

    # Set Options
    Init = Initialization(Step=-1e-2)
    Term = Termination(MaxIter=1000,Tols=[(Domain.f,1e-4*f_0)])
    Repo = Reporting(Requests=[Domain.f, 'Step', 'F Evaluations','Data'])
    Misc = Miscellaneous()
    Options = DescentOptions(Init,Term,Repo,Misc)

    # Print Stats
    PrintSimStats(Domain,Method,Options)

    # Start Solver
    tic = time.time()
    Field_Results = Solve(Start,Method,Domain,Options)
    toc = time.time() - tic

    # Print Results
    PrintSimResults(Options,Field_Results,Method,toc)

    fvals = Field_Results.PermStorage[Domain.f]

    # Plotting
    train_x, train_y = dataset
    if train_x.shape[1] == 2 and Domain.deg == 1:
        xmin = np.min(train_x[:,0])
        xmax = np.max(train_x[:,0])
        ymin = np.min(train_x[:,1])
        ymax = np.max(train_x[:,1])
        dx = xmax - xmin
        dy = ymax - ymin
        xl = xmin - .1*dx
        xr = xmax + .1*dx
        yb = ymin - .1*dy
        yt = ymax + .1*dy
        X,Y = np.meshgrid(np.linspace(xmin,xmax,10,endpoint=True),
                          np.linspace(ymin,ymax,10,endpoint=True))
        coeffs = Field_Results.PermStorage['Data'][-1]
        U = coeffs[0] + coeffs[1]*X + coeffs[2]*Y
        V = coeffs[3] + coeffs[4]*X + coeffs[5]*Y
        A = np.array([[coeffs[1],coeffs[2]],[coeffs[4],coeffs[5]]])
        print(np.linalg.eig(A)[0])
        plt.quiver(X,Y,U,V)
        plt.xlim([xl,xr])
        plt.ylim([yb,yt])
        plt.show()

    plt.plot(fvals)
    plt.show()

    # center = np.array([[0,0,-1],[0,1,0]])
    # Domain.f(center.flatten(),disp=True)
    embed()


if __name__ == '__main__':
    Demo()
