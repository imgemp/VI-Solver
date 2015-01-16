VI-Solver
##A Variational Inequality Solver in Python
===

Ian Gemp
---

##Description
VI-Solver, as the name suggests, is a package that can be used to solve variational inequality problems, VI(F,K), defined as: find x\* such that  \<F(x\*),x-x\*\> >= 0 for all x in K.  This package currently focuses on greedy (local search, gradient methods, etc.) approaches to the problem which take the form x\_k+1 = x\_k + alpha*G(x\_k) where G(x) captures information of F(x) in some way.

##Requirements
This package requires python (2.7 or later) and numpy (1.9.1 or later).

##Usage
There are three main objects that are used to implement the solution to the VI(F,K). One, the domain which provides the mapping F(x).  Two, the solver that performs the update x\_k+1 = x\_k + alpha*G(x\_k).  And three, a storage object that maintains all data that is either required by the solver or requested by the user.

A general template for writing a script is as follows.  Please see the *_Demo.py files for more specific examples.
```python
import time
import numpy as np

from Domains.YourDomain import YourDomain
```
You, the user, creates the file YourDomain.py located in the Domains folder.  This file contains the class YourDomain which has at the very least, an init function to construct the domain and a function F(self,data) to compute and return the mapping F given the data, x.  In this example, YourDomain also has a function, gap, used to judge convergence to the solution x\*.
```python
from Solvers.Euler import Euler
from Projection import IdentityProjection
from Solver import Solve
```
Various VI solvers are available in the Solvers folder - Euler's method is used in this case.  A projection method can also be specified in order to project back onto the feasible set K after an update.  The syntax we use is actually an abuse of notation.  Typically, one would expect the projection operator to be specified as x\_k+1 = P\_K(x\_k).  Instead, we use a different format to allow the possibility of efficient mirror maps between primal and dual spaces (see EntropicProjection).  For this reason, the projection function is expected to take as input the data, x\_k, a stepsize, alpha, and the update direction, F(x\_k) and return x\_k+1.  From Solver we import Solve which is the general method shich drives the updates.
```python
from Options import (
    DescentOptions, Miscellaneous, Reporting, Termination, Initialization)
from Log import PrintSimResults, PrintSimStats
```
Options will be explained in more detail below.  The Log objects are simply used for printing out properties of the experiment being run and the results.
```python
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
