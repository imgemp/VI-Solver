import time
import numpy as np

import matplotlib.pyplot as plt

from VISolver.Domains.AffineGAN import AffineGAN

from VISolver.Solvers.Euler import Euler
from VISolver.Solvers.Extragradient import EG

from VISolver.Projection import BoxProjection
from VISolver.Solver import Solve
from VISolver.Options import (
    DescentOptions, Miscellaneous, Reporting, Termination, Initialization)
from VISolver.Log import PrintSimResults, PrintSimStats
from IPython import embed

def Demo():

    #__AFFINE_GAN__##############################################

    # what happens if we force the diagonal of G to lie in R+?
    # G and -G are both equilibrium solutions -- need to rule one out

    # Define Network and Domain
    mean = np.array([1,2])
    cov = np.array([[1,0.5],[0.5,2]])
    lam, phi = np.linalg.eig(cov)
    mat = np.diag(np.sqrt(lam)).dot(phi)
    # Domain = AffineGAN(u=mean,S=cov,batch_size=1000,alpha=0.,expansion=False)

    # Set Method
    # Method = Euler(Domain=Domain,FixStep=True)
    # Method = EG(Domain=Domain,FixStep=True)

    # Initialize Starting Point
    # Start = np.random.rand(Domain.Dim)
    Start = np.zeros(2*mean.size+cov.size)
    # d0, G0 = Domain.get_args(Start)
    # u0 = G0[:,-1]
    # S0 = G0[:,:Domain.dim]
    # print(u0)
    # print(S0)

    lo = -np.inf*np.ones_like(Start)
    for d in range(mean.size):
        lo[mean.size+d*(mean.size+1)+d] = 0.

    iter_schedule = [2000,4000,4000]
    step_schedule = [1e-1,1e-2,1e-2]
    batch_schedule = [100,100,10]
    Vs = []

    for it,step,batch_size in zip(iter_schedule,step_schedule,batch_schedule):

        Domain = AffineGAN(u=mean,S=cov,batch_size=batch_size,alpha=0.,expansion=False)

        # Set Method
        # Method = Euler(Domain=Domain,FixStep=True,P=BoxProjection(lo=lo))
        Method = EG(Domain=Domain,FixStep=True,P=BoxProjection(lo=lo))

        # Set Options
        Init = Initialization(Step=-step)
        Term = Termination(MaxIter=it)
        Repo = Reporting(Requests=[Domain.V, 'Step', 'F Evaluations',
                                   'Projections','Data'])
        Misc = Miscellaneous()
        Options = DescentOptions(Init,Term,Repo,Misc)

        # Print Stats
        PrintSimStats(Domain,Method,Options)

        # Start Solver
        tic = time.time()
        AG_Results = Solve(Start,Method,Domain,Options)
        toc = time.time() - tic

        # Print Results
        PrintSimResults(Options,AG_Results,Method,toc)

        Vs += AG_Results.PermStorage[Domain.V]
        Start = AG_Results.TempStorage['Data'][-1]

    # d, G = Domain.get_args(data)
    # u = G[:,-1]
    # S = G[:,:Domain.dim]
    # print(u)
    # print(S)
    print(mat)
    # embed()

    for data in AG_Results.PermStorage['Data'][::400]:

        num_samples = 1000
        real, fake = Domain.generate(data,num_samples)

        # u_est = np.mean(fake,axis=0)
        # S_est = np.dot(fake.T,fake)/num_samples
        # print(u_est)
        # print(S_est)

        fig, axs = plt.subplots(2)
        axs[0].scatter(fake[:,0],fake[:,1],s=40, facecolors='none', edgecolors='r',label='fake', zorder=1)
        axs[0].scatter(real[:,0],real[:,1],s=40, marker='*', facecolors='none', edgecolors='k',label='real', zorder=0)
        axs[0].set_title('Comparing Distributions')
        axs[0].set_xlim([-5,10])
        axs[0].set_ylim([-5,10])
        axs[0].legend()
        axs[1].plot(Vs)
        axs[1].set_xlabel('Iteration')
        axs[1].set_ylabel('Minimax Objective (V)')
        plt.show()


if __name__ == '__main__':
    Demo()
