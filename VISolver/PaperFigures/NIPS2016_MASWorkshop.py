'''
This file reproduces the figures (some randomness involved) seen in
"Exploring the Dynamics of Variational Inequality Games with Non-Concave
Utilities" @ NIPS Workshop: Learning, Inference, and Control of Multi-Agent
Systems. 2015
'''

import numpy as np

from VISolver.Domains.CloudServices import (
    CloudServices, CreateNetworkExample)

from VISolver.Solvers.HeunEuler_LEGS import HeunEuler_LEGS

from VISolver.Projection import BoxProjection
from VISolver.Solver import Solve
from VISolver.Options import (
    DescentOptions, Miscellaneous, Reporting, Termination, Initialization)
from VISolver.Log import PrintSimStats

from VISolver.Utilities import ListONP2NP
import time

from VISolver.BoA.Utilities import aug_grid, ind2pt, int2ind, ind2int, pt2inds
from VISolver.BoA.MCGrid_Enhanced import MCT
from VISolver.BoA.Plotting import plotBoA

from VISolver.Plotting import colorline

from matplotlib import pyplot as plt
from matplotlib import cm
import matplotlib as mpl


def plotFigure4(saveFig=False):
    print('Grid Heuristic')

    # Construct grid
    grid = [np.array([0.0,2,21])]*2
    grid = ListONP2NP(grid)
    grid = aug_grid(grid)

    mid = (grid[0,0]+grid[0,1])/2
    inc = grid[0,3]

    # Initialize starting point
    Starts = np.array([[mid,mid+inc],
                       [mid,mid],
                       [mid+6*inc,mid-4*inc]])

    # Record trajectories, LEs, and time steps
    trajs = []
    les = []
    ts = []

    # Stable spiral
    t = np.arange(0,10+.01,.01)
    le = np.array([-1,1j*2*np.pi])
    r = .4
    origin = Starts[0] + [0,r]
    traj = r*np.exp(le[0]*t+le[1]*(t-.25))
    traj = np.vstack([np.real(traj) + origin[0],np.imag(traj) + origin[1]]).T
    trajs += [traj]
    le = np.array([-15,-15])
    les += [le]
    ts += [t]

    # Simple traj
    t = np.arange(0,Starts[1][0]+.01,.01)
    a = np.log(Starts[1][1]+1)/Starts[1][0]
    traj = np.vstack([t,np.exp(a*t)-1]).T
    trajs += [traj[::-1,:]]
    le = np.array([-1,-2])
    les += [le]
    ts += [t]

    # Off-boundary traj
    t = np.arange(0,Starts[2][0]+.01,.01)
    a = np.log(Starts[2][1]+1)/Starts[2][0]
    traj = np.vstack([t,np.exp(a*t)-1]).T
    trajs += [traj[::-1,:]]
    le = np.array([-1,-2])
    les += [le]
    ts += [t]

    # Compute probablity decay of grid points along trajectory
    grid_decay = computeDecay(grid,trajs,les,ts,Starts)
    p = aggregateDecay(grid_decay,grid)

    # Define gray colormap and convert decays to grayscale
    cmap = cm.gray
    norm = mpl.colors.Normalize(vmin=0.,vmax=1.)
    to_rgba = cm.ScalarMappable(norm=norm, cmap=cmap).to_rgba

    adjusted = np.zeros((np.sum(p < 1),2))
    adjcolor = np.zeros((np.sum(p < 1),4))
    shape = tuple(grid[:,2])
    i = 0
    for idp,pi in enumerate(p):
        if pi < 1:
            ind = int2ind(idp,shape)
            adjusted[i,:] = ind2pt(ind,grid)
            adjcolor[i,:] = to_rgba(pi)
            i += 1

    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Plot decayed grid points
    ax.scatter(adjusted[:,0],adjusted[:,1],color=adjcolor)

    # Plot Boundary
    x = np.arange(-1,3,.01)
    ax.plot(x,.25*(x-1)**2+1.05,'k--',lw=2)

    # Plot trajectory
    for idt,traj in enumerate(trajs):
        le = les[idt]
        c = np.max(np.abs(le))
        t = ts[idt]
        T = t[-1]
        z = [np.exp(-c*ti/T) for ti in t]
        colorline(traj[:,0],traj[:,1],z=z,cmap=cmap,norm=norm,linewidth=2)

    # Plot stable endpoints
    plt.scatter(trajs[0][-1][0],trajs[0][-1][1],
                marker='*',s=200,zorder=2,facecolors='w')
    plt.scatter(trajs[1][-1][0],trajs[1][-1][1],
                marker='*',s=200,zorder=2,facecolors='w')

    # Axes properties
    ax.set_xlim([-.1,2.1])
    ax.set_ylim([-.1,2.1])
    minor_ticks = np.arange(-.1,2.1,.1)
    major_ticks = np.arange(0.,2.5,.5)
    ax.set_xticks(major_ticks)
    ax.set_xticks(minor_ticks, minor=True)
    ax.set_yticks(major_ticks)
    ax.set_yticks(minor_ticks, minor=True)
    ax.grid(which='minor', alpha=0.8)
    ax.grid(which='major', alpha=1.0)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    # Annotate LE values
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    Lambda_spi = ','.join('%.1f' % le for le in les[0])
    Lambda_reg = ','.join('%.1f' % le for le in les[1])
    textstr_spi = '$\Lambda=[$'+Lambda_spi+'$]$'
    textstr_reg = '$\Lambda=[$'+Lambda_reg+'$]$'
    ax.text(0.5, 0.95, textstr_spi, transform=ax.transAxes, fontsize=14,
            va='center', ha='center', bbox=props)
    ax.text(0.775, 0.1, textstr_reg, transform=ax.transAxes, fontsize=14,
            va='center', ha='center', bbox=props)

    # Annotate starting points
    ax.text(trajs[0][0,0]-.09, trajs[0][0,1]+.02, '$x_0$', fontsize=18,
            va='center', ha='center', fontweight='bold')
    ax.text(trajs[1][0,0]+.09, trajs[1][0,1]-.02, '$x_1$', fontsize=18,
            va='center', ha='center', fontweight='bold')
    ax.text(trajs[2][0,0]+.09, trajs[2][0,1], '$x_2$', fontsize=18,
            va='center', ha='center', fontweight='bold')

    # Annotate boundary
    ax.annotate('Boundary', xy=(.38, 1.17), xycoords='data',
                xytext=(.3, 1.5), textcoords='data',
                arrowprops=dict(arrowstyle='simple',facecolor='black'),
                ha='center', va='center', size=18,
                )

    # Title and 'square' axes
    fig.suptitle('BoA Algorithm Enhanced with Heuristic',fontsize=18)
    plt.axes().set_aspect('equal')

    if saveFig:
        plt.savefig("Heuristic.png",bbox_inches='tight')
    return fig, ax


def plotFigure5(saveFig=False):
    print('Demand Curve')

    # Define Network and Domain
    Network = CreateNetworkExample(ex=2)
    Domain = CloudServices(Network=Network)
    Domain2 = CloudServices(Network=Network,poly_splice=False)

    # Define parameters for demand function
    i = 0
    j = 3
    pi = np.arange(0,.8,.001)
    pJ = .01
    qi = qJ = 1

    # Compute demand for curve both with and without polynomial splice
    Q = np.zeros_like(pi)
    Q2 = Q.copy()
    t = np.zeros_like(pi)
    for p in range(len(pi)):
        _Q, _t = Domain.Demand_ij(i,j,pi[p],qi,pJ,qJ)
        _Q2, _ = Domain2.Demand_ij(i,j,pi[p],qi,pJ,qJ)
        Q[p] = _Q
        Q2[p] = _Q2
        t[p] = _t

    # Compute critical points: (in)elasticity, splice, zero-utility
    e = pi[np.argmax(t >= 1/np.sqrt(2))]
    Qe = Q[np.argmax(t >= 1/np.sqrt(2))]/12.
    p0 = pi[np.argmax(t >= Domain.t0)]
    Qp0 = Q[np.argmax(t >= Domain.t0)]/12.
    pf = pi[np.argmax(t >= Domain.tf)]
    Qpf = Q[np.argmax(t >= Domain.tf)]/12.

    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Plot demand curve
    Qij, = ax.plot(pi,Q/12.,'k-', linewidth=5)
    exp, = ax.plot(pi,Q2/12.,'k--', linewidth=5)
    ax.plot(p0,Qp0,'ow',markersize=10)
    ax.plot(pf,Qpf,'ow',markersize=10)
    eps = .05
    ax.plot([-eps,.8+eps],[0,0],'k-.')
    ax.plot([0,0],[-eps,1+eps],'k-.')

    # Annotate critical points
    ax.annotate('inelastic', xy=(.04, .97), xycoords='data',
                xytext=(.1, .7), textcoords='data',
                arrowprops=dict(arrowstyle='simple',facecolor='black'),
                ha='center', va='center', size=18,
                )
    ax.annotate('elastic', xy=(e, Qe), xycoords='data',
                xytext=(e+.2, Qe+.2), textcoords='data',
                arrowprops=dict(frac=0.1,headwidth=10,width=4,
                                facecolor='black',shrink=.1),
                ha='center', va='center', size=18,
                )
    ax.annotate('splice', xy=(p0, Qp0), xycoords='data',
                xytext=(p0+.2, Qp0+.2), textcoords='data',
                arrowprops=dict(frac=0.1,headwidth=10,width=4,
                                facecolor='black',shrink=.1),
                ha='center', va='center', size=18,
                )
    ax.annotate('zero-cutoff', xy=(pf, Qpf+.02), xycoords='data',
                xytext=(pf, Qpf+.2*np.sqrt(2)), textcoords='data',
                arrowprops=dict(frac=0.1,headwidth=10,width=4,
                                facecolor='black',shrink=.1),
                ha='center', va='center', size=18,
                )

    ax.set_xlim(-eps,.8+eps)
    ax.set_ylim(-eps,1+eps)

    leg = plt.legend([Qij,exp], [r'$Q_{ij}^{}$', r'$H_{i}e^{-t_{ij}^2}$'],
                     fontsize=20,fancybox=True)
    plt.setp(leg.get_texts()[0], fontsize='20', va='center')
    plt.setp(leg.get_texts()[1], fontsize='20', va='bottom')
    plt.axis('off')

    if saveFig:
        plt.savefig("Demand.png",bbox_inches='tight')
    return fig, ax


def plotFigure6(saveFig=False):
    print('CloudServices BoA')

    msg = 'This method will run for about a week.\n' +\
        'Email imgemp@cs.umass.edu for the results .npy file directly.\n' +\
        'Continue? (y/n) '
    cont = input(msg)
    if cont != 'y':
        return

    # Define Network and Domain
    Network = CreateNetworkExample(ex=2)
    Domain = CloudServices(Network=Network,gap_alpha=2)

    # Set Method
    eps = 1e-2
    Method = HeunEuler_LEGS(Domain=Domain,P=BoxProjection(lo=eps),Delta0=1e0)

    # Set Options
    Init = Initialization(Step=-1e-3)
    Term = Termination(MaxIter=1e5)
    Repo = Reporting(Requests=['Data','Step'])
    Misc = Miscellaneous()
    Options = DescentOptions(Init,Term,Repo,Misc)
    args = (Method,Domain,Options)
    sim = Solve

    # Print Stats
    PrintSimStats(Domain,Method,Options)

    # Construct grid
    grid = [np.array([.5,3.5,6])]*Domain.Dim
    grid = ListONP2NP(grid)
    grid = aug_grid(grid)
    Dinv = np.diag(1./grid[:,3])

    # Compute results
    results = MCT(sim,args,grid,nodes=16,limit=40,AVG=0.00,
                  eta_1=1.2,eta_2=.95,eps=1.,
                  L=16,q=8,r=1.1,Dinv=Dinv)
    ref, data, p, iters, avg, bndry_ids, starts = results

    # Save results
    sim_data = [results,Domain,grid]
    np.save('cloud_'+time.strftime("%Y%m%d_%H%M%S"),sim_data)

    # Plot BoAs
    obs = (4,9)  # Look at new green-tech company
    consts = np.array([3.45,2.42,3.21,2.27,np.inf,.76,.97,.75,1.03,np.inf])
    txt_locs = [(1.6008064516129032, 1.6015625),
                (3.2, 3.2),
                (3.33, 2.53)]
    xlabel = '$p_'+repr(obs[0])+'$'
    ylabel = '$q_'+repr(obs[1])+'$'
    title = 'Boundaries of Attraction for Cloud Services Market'
    fig, ax = plotBoA(ref,data,grid,obs=obs,consts=consts,txt_locs=txt_locs,
                      xlabel=xlabel,ylabel=ylabel,title=title)

    if saveFig:
        plt.savefig('BoA.png',bbox_inches='tight')
    return fig, ax


def plotFigureAppendix(saveFig=False):
    print('Non-Concave Profit Function')

    # Define parameters for profit function
    # Note: d_i is fixed - pi(p,d) = pi(p)
    p = np.arange(0.01,20,.1)
    Q11 = 10.*np.exp(-(p**2./(p+2.))**2.)
    Q12 = np.exp(-(1./10.*p**2./(p+2.))**2.)
    f = (Q11+Q12)*(p-1.)

    fig = plt.figure()
    fig.suptitle('Profit Non-Concave in Price',fontsize=18)
    ax = fig.add_subplot(111)

    ax.plot(p,f,'k',lw=2)

    ax.set_xlim([0,20])
    ax.set_ylim([-2,6])
    plt.xlabel('$p_1$',fontsize=14)
    plt.ylabel('$\pi_1$',fontsize=14)

    if saveFig:
        plt.savefig('NonConcave.png',bbox_inches='tight')
    return fig, ax


# Helper functions
def computeDecay(grid,trajs,les,ts,Starts):
    # Stripped down version of VISolver.BoA.MCGrid_Enhanced.LE
    ddiag = np.linalg.norm(grid[:,3])
    bnd_ind_sum = {}
    for idx,traj in enumerate(trajs):
        le = les[idx]
        c = np.max(np.abs(le))
        t = ts[idx]
        T = t[-1]
        pt0 = Starts[idx]
        cube_inds = pt2inds(pt0,grid)
        cube_pts = np.array([ind2pt(ind,grid) for ind in cube_inds])
        dt = T - t[-2]
        for i, pt in enumerate(traj):
            ti = t[i]
            ds = np.linalg.norm(pt-cube_pts,axis=1)
            if any(ds > ddiag):
                cube_inds = pt2inds(pt,grid)
                cube_pts = np.array([ind2pt(ind,grid) for ind in cube_inds])
                ds = np.linalg.norm(pt-cube_pts,axis=1)
            inbnds = np.all(np.logical_and(cube_pts >= grid[:,0],
                                           cube_pts <= grid[:,1]),
                            axis=1)
            for idx, cube_ind in enumerate(cube_inds):
                if inbnds[idx]:
                    d_fac = 1
                    if not (cube_ind in bnd_ind_sum):
                        bnd_ind_sum[cube_ind] = [0,0]
                    bnd_ind_sum[cube_ind][0] += np.exp(-c*ti/T*d_fac)*dt
                    bnd_ind_sum[cube_ind][1] += dt
    return bnd_ind_sum


def aggregateDecay(bnd_ind_sum,grid):
    # Post processing component of VISolver.BoA.MCGrid_Enhanced.MCT
    shape = tuple(grid[:,2])
    p = np.ones(np.prod(shape))
    bnd_ind_sum_master = {}
    for key,val in bnd_ind_sum.iteritems():
        if not (key in bnd_ind_sum_master):
            bnd_ind_sum_master[key] = [0,0]
        bnd_ind_sum_master[key][0] += val[0]
        bnd_ind_sum_master[key][1] += val[1]
    for key,val in bnd_ind_sum_master.iteritems():
        _int = ind2int(key,shape)
        p[_int] *= val[0]/val[1]
    return p
