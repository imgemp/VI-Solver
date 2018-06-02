import time
import numpy as np
import pandas as pd

import tqdm

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from VISolver.Domains.LQGAN import LQGAN, rand_mu_Sigma

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

# Trial X 'Dimensionality', 'Algorithm', 'Condition #',r'KL/KL_0',r'Euc/Euc_0',r'||F||/||F_0||','Runtime','Steps'
def CompileResults_NdLQGAN(root='lqganresults/'):
    import seaborn as sns
    results = np.load(root+'results.npy').squeeze()
    passed = np.logical_and(res2[:,:,:,:,-1]<10000,~np.isnan(res2[:,:,:,:,-4])).astype('float').mean(axis=(1,2))
    print(passed)
    results = results[:3]
    # dims = np.atleast_2d(np.repeat([1,2,4,6,8,10],10*6)).T
    # trials = np.tile(np.atleast_2d(np.repeat(np.arange(10),6)).T,(6,1))
    # algorithms = np.tile(np.array(['Fcc','Fsim','Feg','Fcon','Freg','EG'])[:,None],(10*6,1))
    # data = np.hstack((dims,trials,algorithms,results.reshape(-1,6)))
    data = results.reshape(-1,10)
    df = pd.DataFrame(data=data,columns=['Dimensionality','Trial','Algorithm','Start','Condition #',r'KL/KL_0',r'Euc/Euc_0',r'||F||/||F_0||','Runtime','Steps'])
    # sns.violinplot(x="Species", y="PetalLengthCm", data=iris, size=6)
    df.replace("nan", np.inf, inplace=True)
    df.replace(inplace=True, to_replace={'Algorithm':dict(zip([0.0,1.0,2.0,3.0,4.0,5.0],['Fcc','Fsim','Feg','Fcon','Freg','EG']))})
    dtypes = ['int','int','str','int','float','float','float','float','float','int']
    for col,dtype in zip(df,dtypes):
        # print(df[col])
        # df.replace("nan", np.inf)
        # print(df[col])
        if dtype != 'str':
            df[col] = pd.to_numeric(df[col])
        df[col] = df[col].astype(dtype)
    # df['Steps'] = np.log(df['Steps'])
    ax = sns.violinplot(x='Dimensionality', y='Steps', hue='Algorithm', data=df, palette="muted", scale="count", inner="stick")
    # ax = sns.violinplot(x='Dimensionality', y='Steps', data=df, palette="muted")
    # ax.set_ylim([0,4])
    # ax.set(yscale="log")
    ax.set_yscale('log')
    fig = ax.get_figure()
    fig.savefig(root+'violin2.png')
    embed()

# res10 contains results for Fcc', Feg' on 1d
# res4 contains results for Fcc,Feg,Fcon for 1d, res5 for 4d

# Fccprime and Fegprime should be run with constant Step=-1e-1 for 1-d problem
# Fcc, Feg, Freg, Fcon all should be run with Delta0=1e-3, MinStep=-1e-2, Step=-1e-4

def Demo_NdLQGAN(root='lqganresults/', Dims=None, Algorithms=None, File=None):
    # Dims = [1]
    Dims = [1,2]
    Nmusigmas = 10
    Lstarts = 10
    Algorithms = dict(zip(['Fsim','EG','Fcon','Freg','Feg','Fcc'],range(6)))
    # Algorithms = dict(zip(['Fcc','Fsim','Feg','Fcon','Freg','EG'],range(6)))
    # Algorithms = dict(zip(['Fcc','Feg','Freg'],range(3)))
    # Algorithms = dict(zip(['Fcc','Feg'],range(2)))
    # Algorithms = dict(zip(['Fcc','Feg','Fccprime','Fegprime'],range(4)))
    # Algorithms = dict(zip(['Fccprime','Fegprime'],range(2)))
    # Algorithms = dict(zip(['Fcc'],range(1)))
    # Algorithms = ['Fcon']
    # Algorithms = dict(zip(['Fcc','Feg','Fcon','Freg','Falt','Funr'],range(6)))

    Iters = 10000

    data = np.empty((len(Dims),Nmusigmas,Lstarts,len(Algorithms),10))

    for i, dim in enumerate(tqdm.tqdm(Dims)):
        Domain = LQGAN(dim=dim, var_only=True)

        # Dimensionality
        s = Domain.s
        pdim = Domain.Dim

        # Set Constraints
        loA = -np.inf*np.ones((dim,dim))
        loA[range(dim),range(dim)] = 1e-2
        lo = np.hstack(([-np.inf]*(dim+s), loA[np.tril_indices(dim)], [-np.inf]*dim))
        P = BoxProjection(lo=lo)

        for musigma in tqdm.tqdm(range(Nmusigmas)):
            # Reset LQGAN to random mu and Sigma
            mu, Sigma = rand_mu_Sigma(dim=dim)
            # print(Sigma.flatten().item())
            mu = np.zeros(dim)
            # Sigma = np.array([[1]])
            # Sigma = np.diag(np.random.rand(dim)*10+1.)
            lams = np.linalg.eigvals(Sigma)
            K = lams.max()/lams.min()
            # print(np.linalg.eigvals(Sigma))
            Domain.set_mu_sigma(mu=mu,sigma=Sigma)
            # print(mu,Sigma)

            for l in tqdm.tqdm(range(Lstarts)):
                # Intialize G and D variables to random starting point
                # should initialize A to square root of random Sigma
                # Start = P.P(10*np.random.rand(pdim)-5.)
                Start = np.zeros(pdim)
                Start[:s] = np.random.rand(s)*10-5
                Start[s:s+dim] = 0.  # set w_1 to zero
                Start[-dim:] = mu  # set b to mu
                Start[-dim-s:-dim] = np.linalg.cholesky(rand_mu_Sigma(dim=dim)[1])[np.tril_indices(dim)]
                Start = P.P(Start)

                # Calculate Initial KL and Euclidean distance
                KL_0 = Domain.dist_KL(Start)
                Euc_0 = Domain.dist_Euclidean(Start)
                norm_F_0 = Domain.norm_F(Start)
                # print(KL_0,Euc_0,norm_F_0)

                for j, alg in enumerate(tqdm.tqdm(Algorithms)):

                    # Step = -1e-3
                    if alg == 'EG':
                        Step = -1e-3
                        Domain.preconditioner = 'Fsim'
                        Method = EG(Domain=Domain,FixStep=True,P=P)
                    elif alg == 'Fsim':
                        Step = -1e-4
                        Domain.preconditioner='Fsim'
                        Method = HeunEuler_PhaseSpace(Domain=Domain,Delta0=1e-3,P=P,MinStep=-1e-2)
                        # Method = Euler(Domain=Domain,FixStep=True,P=P)
                    elif alg == 'Fcon':
                        Step = -1e-4
                        Domain.preconditioner = alg
                        Method = HeunEuler_PhaseSpace(Domain=Domain,Delta0=1e-3,P=P,MinStep=-1e-2)
                        # Method = Euler(Domain=Domain,FixStep=True,P=P)
                    elif alg == 'Falt' or alg == 'Funr':
                        Step = -5e-3
                        Domain.preconditioner = alg
                        # Method = HeunEuler_PhaseSpace(Domain=Domain,Delta0=1e-5,P=P)
                        Method = Euler(Domain=Domain,FixStep=True,P=P)
                    elif alg == 'Fccprime' or alg == 'Fegprime':
                        Step = -1e-1
                        Domain.preconditioner = alg
                        Method = Euler(Domain=Domain,FixStep=True,P=P)
                    else:
                        # Step = -1e-1
                        Step = -1e-4
                        Domain.preconditioner = alg
                        # Method = HeunEuler_PhaseSpace(Domain=Domain,Delta0=1e-0,P=P,MinStep=-10.)  # for 2d+
                        # Method = HeunEuler_PhaseSpace(Domain=Domain,Delta0=1e-2,P=P,MinStep=-1e-1) # 13 slow
                        Method = HeunEuler_PhaseSpace(Domain=Domain,Delta0=1e-3,P=P,MinStep=-1e-2) # slow
                        # Method = HeunEuler_PhaseSpace(Domain=Domain,Delta0=1e-1,P=P,MinStep=-1e-1) better
                        # Method = HeunEuler_PhaseSpace(Domain=Domain,Delta0=1e-1,P=P,MinStep=-1e-10) slow
                        # Method = HeunEuler_PhaseSpace(Domain=Domain,Delta0=1e-1,P=P,MinStep=-1e-2) best so far, Feg is not working well?
                        # Method = HeunEuler_PhaseSpace(Domain=Domain,Delta0=1e-2,P=P,MinStep=-1e-2) ok
                        # Method = HeunEuler_PhaseSpace(Domain=Domain,Delta0=1e-2,P=P,MinStep=-1e-1)
                        # Delta0=1e-2,MinStep=-1e-1 for speed on 1d, scaled versions are surprisingly worse
                    
                    # Set Options
                    Init = Initialization(Step=Step)
                    Tols = [#(Domain.norm_F,1e-3*norm_F_0),
                            # (Domain.dist_KL,1e-3*KL_0),
                            (Domain.dist_Euclidean,1e-3*Euc_0),
                            (Domain.isNotNaNInf,False)]
                    Term = Termination(MaxIter=Iters,Tols=Tols,verbose=False)
                    Repo = Reporting(Requests=[Domain.norm_F, Domain.dist, 
                                               Domain.dist_KL, Domain.dist_Euclidean,
                                               Domain.isNotNaNInf])
                    # 'Step','Data',
                    Misc = Miscellaneous()
                    Options = DescentOptions(Init,Term,Repo,Misc)

                    # Start Solver
                    tic = time.time()
                    LQ_Results = Solve(Start,Method,Domain,Options)
                    toc = time.time() - tic

                    KL = LQ_Results.PermStorage[Domain.dist_KL][-1]
                    Euc = LQ_Results.PermStorage[Domain.dist_Euclidean][-1]
                    norm_F = LQ_Results.PermStorage[Domain.norm_F][-1]
                    # x = np.array(LQ_Results.PermStorage['Data'])
                    # Steps = np.array(LQ_Results.PermStorage['Step'])
                    runtime = toc
                    steps = LQ_Results.thisPermIndex
                    # embed()

                    data[i,musigma,l,j,:] = np.array([dim,musigma,Algorithms[alg],l,K,KL/KL_0,Euc/Euc_0,norm_F/norm_F_0,runtime,steps])
                    # embed()
            np.save(root+'results15.npy', data)


def Demo_1dLQGAN(root='figures/'):
    ##############################################################################
    # Compre Trajectories Plot ###################################################
    ##############################################################################

    # Define Network and Domain
    dim = 1
    s = (dim**2+dim)//2
    mu = np.array([0])
    sig = np.array([[1]])

    np.set_printoptions(linewidth=200)

    # Set Constraints
    loA = -np.inf*np.ones((dim,dim))
    loA[range(dim),range(dim)] = 1e-2
    lo = np.hstack(([-np.inf]*(dim+s), loA[np.tril_indices(dim)], [-np.inf]*dim))
    P = BoxProjection(lo=lo)

    xoff = 0
    yoff = 0
    scale = 30

    datas = []
    dists = []
    methods = ['Fcc','Fsim','Feg','Fcon','Freg','EG','Falt','Funr']
    for method in methods:

        if method == 'EG':
            Step = -1e-2
            Iters = 10000
            Domain = LQGAN(mu=mu,sig=sig,preconditioner='Fsim')
            Method = EG(Domain=Domain,FixStep=True,P=P)
        elif method == 'Fsim':
            Step = -1e-3
            Iters = 100000
            Domain = LQGAN(mu=mu,sig=sig,preconditioner='Fsim')
            Method = Euler(Domain=Domain,FixStep=True,P=P)
            # Method = HeunEuler_PhaseSpace(Domain=Domain,Delta0=1e-5,MinStep=-1.,P=P)
        elif method == 'Fcon':
            Step = -1e-6
            # Iters = 2750
            Iters = 3000
            Domain = LQGAN(mu=mu,sig=sig,preconditioner=method)
            Method = HeunEuler_PhaseSpace(Domain=Domain,Delta0=1e-5,P=P)
        elif method == 'Falt' or method == 'Funr':
            Step = -5e-3
            # Iters = 2750
            Iters = 10000
            Domain = LQGAN(mu=mu,sig=sig,preconditioner=method)
            # Method = HeunEuler_PhaseSpace(Domain=Domain,Delta0=1e-5,P=P)
            Method = Euler(Domain=Domain,FixStep=True,P=P)
        else:
            Step = -1e-5
            Iters = 10000
            Domain = LQGAN(mu=mu,sig=sig,preconditioner=method)
            Method = HeunEuler_PhaseSpace(Domain=Domain,Delta0=1e-5,P=P)
        
        # Initialize Starting Point
        Start = np.array([50.,0.,30.,0.])

        # Set Options
        Init = Initialization(Step=Step)
        Term = Termination(MaxIter=Iters,Tols=[(Domain.dist_Euclidean,1e-4)])
        Repo = Reporting(Requests=['Step', 'F Evaluations',
                                   'Projections','Data',Domain.dist_Euclidean])
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

        datas += [np.array(LQ_Results.PermStorage['Data'])]
        dists += [np.array(LQ_Results.PermStorage[Domain.dist_Euclidean])]

    X, Y = np.meshgrid(np.linspace(-2*scale + xoff, 2*scale + xoff, 21), np.arange(1e-2 + yoff, 4*scale + yoff, .2*scale))
    U = np.zeros_like(X)
    V = np.zeros_like(Y)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            vec = -Domain._F(np.array([X[i,j],0.,Y[i,j],0.]))
            U[i,j] = vec[0]
            V[i,j] = vec[2]

    fs = 18
    fig = plt.figure()
    ax = fig.add_subplot(111)
    Q = plt.quiver(X[::3, ::3], Y[::3, ::3], U[::3, ::3], V[::3, ::3],
                   pivot='mid', units='inches')
    colors = ['r','c','b','k','g','m','gray','orange']
    methods = [r'$F_{cc}$',r'$F$',r'$F_{eg}$',r'$F_{con}$',r'$F_{reg}$',r'$EG$',r'$F_{alt}$',r'$F_{unr}$']
    for data, color, method in zip(datas,colors,methods):
        if method == 'EG' or method == 'Funr':
            ax.plot(data[:,0],data[:,2],'-.',color=color,label=method,zorder=0)
        else:
            ax.plot(data[:,0],data[:,2],color=color,label=method,zorder=0)
    ax.plot([data[0,0]],[data[0,2]],'k*')
    ax.plot(mu,sig[0],'*',color='gold')
    # ax.set_xlim([-2*scale + xoff,2*scale + xoff])
    # ax.set_ylim([-.1*scale + yoff,4*scale + yoff])
    ax.set_xlim([-60., 60.])
    ax.set_ylim([-.1*scale + yoff, 100.])
    ax.set_xlabel(r'$w_2$', fontsize=fs)
    ax.set_ylabel(r'$a$', fontsize=fs)
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(14)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(14)
    plt.title('Trajectories for Various Equilibrium Algorithms', fontsize=16)
    # plt.legend(fontsize=fs, loc="upper left")

    fs = 24
    # locs = [(15,25),(-35,80),(0,40),(42.5,3.0),(20,2.5),(-30,35),(-40,20),(-20,20)]
    # locs = [(15,25),(-35,80),(2.5,40),(42.5,3.0),(20,2.5),(-45,45),(-42.5,5),(-22.5,5)]
    locs = [(15,25),(-35,75),(2.5,40),(42.5,3.0),(20,2.5),(-45,45),(-42.5,5),(-22.5,5)]
    for method,color,loc in zip(methods,colors,locs):
        ax.annotate(method, xy=loc, color=color, zorder=1, fontsize=fs,
                    bbox=dict(facecolor='white', edgecolor='white', pad=0.0))

    plt.savefig(root+'trajcomp.png',bbox_inches='tight')

    ##############################################################################
    # Euclidean Distance to Equilibrium vs Iterations ############################
    ##############################################################################

    # Old Version of Plot
    # # fig = plt.figure()
    # # ax = fig.add_subplot(111)
    # # colors = ['r','gray','b','k','g']
    # # for dist, color, method in zip(dists,colors,methods):
    # #     ax.plot(dist,color=color,label=method)
    # # ax.set_xlabel('Iterations')
    # # ax.set_ylabel('Distance to Equilibrium')
    # # plt.legend()
    # # plt.title('Iteration vs Distance to Equilibrium for Various Equilibrium Algorithms')
    # # plt.savefig('runtimecomp.png')

    # More Recent Version of Plot
    fig,(ax,ax2) = plt.subplots(1,2,sharey=True, facecolor='w')
    for dist, color, method in zip(dists,colors,methods):
        # if method == 'simGD':
        iters = np.arange(dist.shape[0])/1000
        # print(dist[:5])
        ax.plot(iters,dist,color=color,label=method)
        ax2.plot(iters,dist,color=color,label=method)
        # else:
        #     ax.plot(dist,color=color,label=method)
    ax.set_xlim(-.1,10)
    ax.set_xticks([0,2,4,6,8,10])
    ax2.set_xlim(50,100)
    ax.set_ylim(-5,95)
    ax2.set_ylim(-5,95)

    # hide the spines between ax and ax2
    ax.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax.yaxis.tick_left()
    ax.tick_params(labelright='off')
    ax2.yaxis.tick_right()
    ax2.set_xticks([75,100])

    d = .015 # how big to make the diagonal lines in axes coordinates
    # arguments to pass plot, just so we don't keep repeating them
    kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
    ax.plot((1-d,1+d), (-d,+d), **kwargs)
    ax.plot((1-d,1+d),(1-d,1+d), **kwargs)

    kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
    ax2.plot((-d,+d), (1-d,1+d), **kwargs)
    ax2.plot((-d,+d), (-d,+d), **kwargs)

    # ax.set_xlabel('Iterations')
    fig.text(0.5, 0.02, 'Thousand Iterations', ha='center', fontsize=12)
    ax.set_ylabel('Distance to Equilibrium',fontsize=12)
    # ax.legend()
    ax2.legend(fontsize=18)
    plt.suptitle('Iteration vs Distance to Equilibrium for Various Equilibrium Algorithms')
    plt.savefig(root+'runtimecomp.png')

    ##############################################################################
    # Individual Trajectory Plots ################################################
    ##############################################################################

    # Set Constraints
    loA = -np.inf*np.ones((dim,dim))
    loA[range(dim),range(dim)] = 1e-4
    lo = np.hstack(([-np.inf]*(dim+s), loA[np.tril_indices(dim)], [-np.inf]*dim))
    P = BoxProjection(lo=lo)

    xlo, xhi = -5, 5
    ylo, yhi = 0, 2
    methods = ['Fcc','Feg','Fcon','Freg','Falt','Funr']
    colors = ['r','b','k','g','gray','orange']
    for method, color in zip(methods,colors):
        Step = -1e-5
        Iters = 10000
        Domain = LQGAN(mu=mu,sig=sig,preconditioner=method)
        Method = HeunEuler_PhaseSpace(Domain=Domain,Delta0=1e-5,P=P)
        # Iters = 10000
        # Method = Euler(Domain=Domain,FixStep=True)
        
        # Initialize Starting Point
        Start = np.array([3.,0.,0.2,0.])

        # Set Options
        Init = Initialization(Step=Step)
        Term = Termination(MaxIter=Iters,Tols=[(Domain.dist_Euclidean,1e-4)])
        Repo = Reporting(Requests=['Step', 'F Evaluations',
                                   'Projections','Data',Domain.dist_Euclidean])
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
        
        X, Y = np.meshgrid(np.linspace(xlo, xhi, 50), np.linspace(ylo, yhi, 50))
        U = np.zeros_like(X)
        V = np.zeros_like(Y)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                vec = -Domain.F(np.array([X[i,j],0.,Y[i,j],0.]))
                U[i,j] = vec[0]
                V[i,j] = vec[2]

        fig = plt.figure()
        ax = fig.add_subplot(111)
        Q = plt.quiver(X[::3, ::3], Y[::3, ::3], U[::3, ::3], V[::3, ::3],
                       pivot='mid', units='inches')
        ax.plot(data[:,0],data[:,2],color=color,label=method)
        ax.plot([data[0,0]],[data[0,2]],'k*')
        ax.plot(mu,sig[0],'c*')
        ax.set_xlim([xlo, xhi])
        ax.set_ylim([0, yhi])
        ax.set_xlabel(r'$w_2$')
        ax.set_ylabel(r'$a$')
        plt.title('Dynamics for '+method)
        plt.savefig(root+method+'_dyn')

def MonotoneRegion(root='figures/'):
    matplotlib.rcParams['xtick.direction'] = 'out'
    matplotlib.rcParams['ytick.direction'] = 'out'

    delta = 0.025
    w2 = np.arange(-5.0, 5.0, delta)
    a = np.arange(0.0, 5.0, delta)
    W2, A = np.meshgrid(w2, a)
    LAM_cc = -1. + 5.*(A**2.) - np.sqrt(1. + A**4. + 2.*(A**2.)*(-1. + 8.*(W2**2.)))  # They're actually the same boundary!!!!
    # solve for when square root above is less than b^2 and you get same as below, LAM_eg
    LAM_eg = -1. + 3.*(A**2.) - 2.*(W2**2.)
    # You can force all the contours to be the same color.
    plt.figure()
    CS = plt.contour(W2, A, LAM_cc, 10,
                     colors='k',  # negative contours will be dashed by default
                     )
    plt.clabel(CS, fontsize=9, inline=1)
    plt.title('Monotone Region: Crossing the Curl')
    plt.savefig(root+'monregion_cc.png')
    plt.close()

    plt.figure()
    CS = plt.contour(W2, A, LAM_eg, 10,
                     colors='k',  # negative contours will be dashed by default
                     )
    plt.clabel(CS, fontsize=9, inline=1)
    plt.title('Monotone Region: Extragradient')
    plt.savefig(root+'monregion_eg.png')
    plt.close()

def plot_w1b(root='figures/'):
    matplotlib.rcParams['xtick.direction'] = 'out'
    matplotlib.rcParams['ytick.direction'] = 'out'

    plt.figure()
    r = 0.5
    X, Y = np.meshgrid(np.linspace(-r, r, 10), np.linspace(-r, r, 10))
    # U = np.zeros_like(X)
    # V = np.zeros_like(Y)
    # for i in range(X.shape[0]):
    #     for j in range(X.shape[1]):
    #         U[i,j] = -Y[i,j]
    #         V[i,j] = X[i,j]
    U = Y
    V = -X

    fig = plt.figure()
    ax = fig.add_subplot(111)
    skip = 1
    Q = plt.quiver(X[::skip, ::skip], Y[::skip, ::skip], U[::skip, ::skip], V[::skip, ::skip],
                   pivot='mid', units='inches')
    v0 = np.array([0,r*0.6])
    EGv0 = np.array([v0[1],-v0[0]])
    F0 = v0 - EGv0
    v1 = v0 - 0.5*EGv0
    EGv1 = np.array([v1[1],-v1[0]])
    F1 = v1 - EGv1
    v2 = v1 - 0.5*EGv1
    v1actual = v0 - 0.5*EGv1
    # ax.plot([v0[0]],[v0[1]],'r*')
    # ax.plot([v0[0],v1[0]],[v0[1],v1[1]],'r-.')
    ax.arrow(v0[0], v0[1], -.5*EGv0[0], -.5*EGv0[1], head_width=0.05, head_length=0.05, fc='w', ec='r', ls='-', zorder=5)
    # ax.plot([v1[0],v2[0]],[v1[1],v2[1]],'r-.')
    ax.arrow(v1[0], v1[1], -.5*EGv1[0], -.5*EGv1[1], head_width=0.05, head_length=0.05, fc='w', ec='r', ls='-', zorder=4)
    # ax.plot([v0[0],v1actual[0]],[v0[1],v1actual[1]],'r-')
    ax.arrow(v0[0], v0[1], -.5*EGv1[0], -.5*EGv1[1], head_width=0.05, head_length=0.05, fc='r', ec='r', ls='-', zorder=4)
    ax.plot([v0[0]],[v0[1]],'r*', markersize=8)
    # ax.plot([0.],[0.],'*', markersize=10)
    ax.plot([0.],[0.],linestyle="None",marker=(5,1,0),markersize=20,color='gold')

    # plt.rc('text', usetex=True)
    # plt.rc('font', family='serif')
    fs = 24
    ax.annotate(r'$x_k$', xy=(v0[0]+.03, v0[1]-.04), color='blue', zorder=1, fontsize=fs,
                bbox=dict(facecolor='white', edgecolor='white', pad=0.0))
    ax.annotate(r'$\hat{x}_{k+1}$', xy=(v1[0]-.13, v1[1]+.06), color='blue', zorder=1, fontsize=fs,
                bbox=dict(facecolor='white', edgecolor='white', pad=0.0))
    # ax.annotate(r'$x_{k+1}$', xy=(v1actual[0]-.09, v1actual[1]-.11), color='blue', zorder=1, fontsize=fs,
    #             bbox=dict(facecolor='white', edgecolor='white', pad=0.0))
    ax.annotate(r'$x^{eg}_{k+1}$', xy=(v1actual[0]-.09, v1actual[1]-.13), color='blue', zorder=1, fontsize=fs,
                bbox=dict(facecolor='white', edgecolor='white', pad=0.0))
    # ax.axis('equal')
    # ax.set_xlim([-r, r])
    # ax.set_ylim([-r, r])
    ax.set_aspect('equal', 'box')
    ax.set_xlabel(r'$w_1$', fontsize=fs)
    ax.set_ylabel(r'$b$', fontsize=fs)
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(20)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(20)
    # plt.tight_layout()
    # fig.tight_layout()
    # plt.title('Dynamics for '+method)

    v0 = np.array([0,-r*0.6])
    Fv0 = np.array([v0[1],-v0[0]])
    CCv0 = np.array([v0[0],v0[1]])
    Fv1 = v0 - Fv0
    CCv1 = v0 - CCv0
    ax.arrow(v0[0], v0[1], -.5*Fv0[0], -.5*Fv0[1], head_width=0.05, head_length=0.05, fc='r', ec='r', ls='-', zorder=5)
    ax.arrow(v0[0], v0[1], -.5*CCv0[0], -.5*CCv0[1], head_width=0.05, head_length=0.05, fc='r', ec='r', ls='-', zorder=4)
    
    # plt.rc('text', usetex=True)
    # plt.rc('font', family='serif')
    fs = 24
    ax.plot([v0[0]],[v0[1]],'r*', markersize=8)
    ax.annotate(r'$x_k$', xy=(v0[0]-.12, v0[1]-.06), color='blue', zorder=1, fontsize=fs,
                bbox=dict(facecolor='white', edgecolor='white', pad=0.0))
    ax.annotate(r'$x_{k+1}$', xy=(Fv1[0]-.09, Fv1[1]-.04), color='blue', zorder=1, fontsize=fs,
                bbox=dict(facecolor='white', edgecolor='white', pad=0.0))
    ax.annotate(r'$x^{cc}_{k+1}$', xy=(CCv1[0]+.04, CCv1[1]-.13), color='blue', zorder=1, fontsize=fs,
                bbox=dict(facecolor='white', edgecolor='white', pad=0.0))

    plt.savefig(root+'w1b_subsystem.png',bbox_inches='tight')


if __name__ == '__main__':
    Demo_NdLQGAN()
    # CompileResults_NdLQGAN()
    # Demo_1dLQGAN()
    # MonotoneRegion()
    # plot_w1b()
