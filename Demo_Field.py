import time
import numpy as np

from VISolver.Domains.FieldRegressor import FieldRegressor
from VISolver.Domains.PolyRegressor import PolyRegressor

from VISolver.Solvers.HeunEuler import HeunEuler

from VISolver.Solver import Solve
from VISolver.Options import (
    DescentOptions, Miscellaneous, Reporting, Termination, Initialization)
from VISolver.Log import PrintSimResults, PrintSimStats


def Demo():

    #__FIELD_REGRESSION__##############################################

    # Construct Dataset
    N = 10
    train_x = np.random.rand(N,2)
    train_y = np.random.rand(N)
    dataset = (train_x,train_y)

    # Define Regressor
    Domain = FieldRegressor(dataset,deg=2)

    # Set Method
    Method = HeunEuler(Domain=Domain,Delta0=1e-2)

    # Initialize Starting Point
    Start = np.ones(Domain.Dim)

    # Calculate Initial Error
    f_0 = Domain.f(Start)

    # Set Options
    Init = Initialization(Step=-1e-2)
    Term = Termination(MaxIter=1000,Tols=[(Domain.f,1e-2*f_0)])
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

    #__POLY_REGRESSION__##############################################

    # Define Regressor
    Domain = PolyRegressor(dataset,deg=3)

    # Set Method
    Method = HeunEuler(Domain=Domain,Delta0=1e-2)

    # Initialize Starting Point
    Start = np.ones(Domain.Dim)

    # Calculate Initial Error
    f_0 = Domain.f(Start)

    # Set Options
    Init = Initialization(Step=-1e-2)
    Term = Termination(MaxIter=1000,Tols=[(Domain.f,1e-2*f_0)])
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

if __name__ == '__main__':
    Demo()
