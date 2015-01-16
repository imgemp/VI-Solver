VI-Solver
##A Variational Inequality Solver in Python
===

Ian Gemp
---

##Description
VI-Solver, as the name suggests, is a package that can be used to solve variational inequality problems, VI(F,K), defined as: find x * such that  <F(x *),x-x *> >= 0 for all x in K.  This package currently focuses on greedy (local search, gradient methods, etc.) approaches to the problem which take the form x_k+1 = x_k + alpha *G(x_k) where G(x) captures information of F(x) in some way.

##Requirements
This package requires python (2.7 or later) and numpy (1.9.1 or later).

##Usage
There are three main objects that are used to implement the solution to the VI(F,K). One, the domain which provides the mapping F(x).  Two, the solver that performs the update x_k+1 = x_k + alpha*G(x_k).  And three, a storage object that maintains all data that is either required by the solver or requested by the user.

A general template for running the code is as follows.  Please see the *_Demo.py files for more specific examples.
```python
import time
import numpy as np

from Domains.YourDomain import YourDomain

from Solvers.Euler import Euler

from Projection import IdentityProjection
from Solver import Solve
from Options import (
    DescentOptions, Miscellaneous, Reporting, Termination, Initialization)
from Log import PrintSimResults, PrintSimStats


def Demo():

    Domain = YourDomain(param1=0,param2=1,param3=3)

    # Set Method
    Method = Euler(Domain=Domain,P=IdentityProjection())

    # Initialize Starting Point
    Start = np.zeros(Domain.Dim)

    # Calculate Initial Gap
    gap_0 = Domain.gap(Start)

    # Set Options
    Init = Initialization(Step=-0.1)
    Term = Termination(MaxIter=1000,Tols=[(Domain.gap,1e-3*gap_0)])
    Repo = Reporting(Requests=[Domain.gap, 'Step', 'F Evaluations',
                               'Projections'])
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

if __name__ == '__main__':
  Demo()
```
