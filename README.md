#VI-Solver
*A Variational Inequality Solver in Python*

##Description
VI-Solver, as the name suggests, is a package that can be used to solve variational inequality problems, VI(F,K), defined as: find x\* such that  \<F(x\*),x-x\*\> >= 0 for all x in K.  This package currently focuses on greedy (local search, gradient methods, etc.) approaches to the problem which take the form x\_k+1 = x\_k + alpha*G(x\_k) where G(x) captures information of F(x) in some way.

##Requirements
This package requires python (2.7 or later) and numpy (1.9.1 or later).

##Domains
A number of domains are included in the VISolver python package including:

*Networks*

- BloodBank: A blood banking network as described in [1].

*Optimization*

- Sphere: Simply represents L2 distance from the origin.  Minimum is the origin.
- Rosenbrock: A non-convex objective function with a global minimum for performance tests.

*Standard VI Domains* (as described in [2])

- Kojima Shindo: F: R4 --> R4
- Watson: F: R10 --> R10
- Sun: F: Rn --> Rn, F(x) = Ax + b
- Modified HP Hard (MHPH): F: Rn --> Rn, F(x) = Ax + b, A is randomly generated as M*M' (VI is monotone)
- Randomly Generated (RG): F: Rn --> Rn, F(x) = Ax + b, A is uniformly generated (monotinicity of VI not guaranteed)

[1] Nagurney & Masoumi 2012. Supply Chain Network Design of a Sustainable Blood Banking System.

[2] Dang & Lan 2013. On the Convergence Properties of Non-Euclidean Extragradient Methods for Variational Inequalities with Generalized Monotone Operators.

##Demos
A few demos are included in the package.

- Demo_BloodBank.py
- Demo_DangLan.py
- Demo_Rosenbrock.py

To run one of the demos after cloning this repository, open up a command line interface (Terminal on a Mac) and enter the following commands.

```bash
cd [path to your local cloned repository (e.g. /Users/username/Desktop/VI-Solver)]
python Demo_[e.g. BloodBank].py
```

That's it! Assuming you have PYTHON setup correctly and your PYTHON path is set, the Demo should run and print alerts and results out to the command line.

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
Various VI solvers are available in the Solvers folder - Euler's method is used in this case.  A projection method can also be specified in order to project back onto the feasible set K after an update.  The syntax we use is actually an abuse of notation.  Typically, one would expect the projection operator to be specified as x\_k+1 = P\_K(x\_k).  Instead, we use a different format to allow the possibility of efficient mirror maps between primal and dual spaces (see EntropicProjection).  For this reason, the projection function is expected to take as input the data, x\_k, a stepsize, alpha, and the update direction, F(x\_k) and return x\_k+1.  From Solver we import Solve which is the general method that drives the updates.
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
```
To start, the user initializes their domain according to how they defined their constructor.  Next, the method is initialized by passing both the domain (which contains the function defining the F mapping) and specific projection operator to be used (IdentityProjection by default).  In this example, we calculate the initial value of the gap function for use later.
```python
    # Set Options
    Init = Initialization(Step=-0.1)
    Term = Termination(MaxIter=1000,Tols=[(Domain.gap,1e-3*gap_0)])
    Repo = Reporting(Requests=[Domain.gap, 'Step', 'F Evaluations',
                               'Projections'])
    Misc = Miscellaneous()
    Options = DescentOptions(Init,Term,Repo,Misc)
```
- Initialization: As of now, it only sets the initial stepsize.
- Termination: This is used to specifiy convergence criteria.  MaxIter is required, however, additional tolerances can be specified as a list of (criterium,<=value) tuples.  The criteria can be anything that is tracked by the solver itself or has been requested to be tracked by the user.  Here, we specify an additional convergence criterium in which the solver will terminate if the gap function falls below 1e-3*gap_0.
- Reporting: Here, the user may specify information that they wish to be tracked throughout the solver's approach to the solution.  These can either be pieces of information that are tracked by the solver itself (see the solver's TempStorage object) or any other additional information that may be reported by the Domain object (such as a gap function).
- Miscellaneous: As of now, this only sets the minimum of a function, f, assuming F is the gradient of f and is only used when the VI has an equivalent optimization formulation.  This has been left blank in this example.
The entire options object is constructed using each of the predefined options above.
```python
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
The rest of the demo is straightforward.  For convenience, PrintSimStats prints out various information about the current experiment.  A timer is then started, afterwhich, the Solve mechanism begins iteratively cranking away at the solution to the VI using the prescribed starting point, x\_0, as well as the defined method, domain, and options.  Upon completion, information on the results is printed.
