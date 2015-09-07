import time
import numpy as np

from VISolver.Utilities import (
    ListONP2NP, aug_grid, ind2int, ind2pt2, pt2inds2, int2ind)

from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib.collections import LineCollection

from IPython import embed


def groupstuff(grid,trajs,lams,ts,Starts):
    ddiag = np.linalg.norm(grid[:,3])
    # dmax = ddiag*.5
    bnd_ind_sum = {}
    for idx,traj in enumerate(trajs):
        lam = lams[idx]
        c = np.max(np.abs(lam))
        t = ts[idx]
        T = t[-1]
        pt0 = Starts[idx]
        cube_inds = pt2inds2(pt0,grid)
        cube_pts = np.array([ind2pt2(ind,grid) for ind in cube_inds])
        dt = T - t[-2]
        for i, pt in enumerate(traj):
            ti = t[i]
            ds = np.linalg.norm(pt-cube_pts,axis=1)
            if any(ds > ddiag):
                cube_inds = pt2inds2(pt,grid)
                cube_pts = np.array([ind2pt2(ind,grid) for ind in cube_inds])
                ds = np.linalg.norm(pt-cube_pts,axis=1)
            inbnds = np.all(np.logical_and(cube_pts >= grid[:,0],
                                           cube_pts <= grid[:,1]),
                            axis=1)
            for idx, cube_ind in enumerate(cube_inds):
                if inbnds[idx]:
                    # d = ds[idx]
                    # d_fac = max(1-d/dmax,0)
                    d_fac = 1
                    if not (cube_ind in bnd_ind_sum):
                        bnd_ind_sum[cube_ind] = [0,0]
                    bnd_ind_sum[cube_ind][0] += np.exp(-c*ti/T*d_fac)*dt
                    bnd_ind_sum[cube_ind][1] += dt
    return bnd_ind_sum


def procgroup(bnd_ind_sum,grid):
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

def make_segments(x, y):
    '''
    Create list of line segments from x and y coordinates, in the correct format for LineCollection:
    an array of the form   numlines x (points per line) x 2 (x and y) array
    '''

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    return segments


# Interface to LineCollection:

def colorline(x, y, z=None, cmap=plt.get_cmap('copper'), norm=plt.Normalize(0.0, 1.0), linewidth=3, alpha=1.0):
    '''
    Plot a colored line with coordinates x and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    '''

    # Default colors equally spaced on [0,1]:
    if z is None:
        z = np.linspace(0.0, 1.0, len(x))

    # Special case if a single number:
    if not hasattr(z, "__iter__"):  # to check for numerical input -- this is a hack
        z = np.array([z])

    z = np.asarray(z)

    segments = make_segments(x, y)
    lc = LineCollection(segments, array=z, cmap=cmap, norm=norm, linewidth=linewidth, alpha=alpha)

    ax = plt.gca()
    ax.add_collection(lc)

    return lc


def Demo():

    grid = [np.array([0.0,2,21])]*2
    grid = ListONP2NP(grid)
    grid = aug_grid(grid)
    Dinv = np.diag(1./grid[:,3])

    mid = (grid[0,0]+grid[0,1])/2
    inc = grid[0,3]

    Starts = np.array([[mid,mid+inc],
                       [mid,mid],
                       [mid+6*inc,mid-4*inc]])
    trajs = []
    lams = []
    ts = []

    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Stable spiral
    t = np.arange(0,10+.01,.01)
    lam = np.array([-1,1j*2*np.pi])
    r = .4
    origin = Starts[0] + [0,r]
    traj = r*np.exp(lam[0]*t+lam[1]*(t-.25))
    traj = np.vstack([np.real(traj) + origin[0],np.imag(traj) + origin[1]]).T
    trajs += [traj]
    lam = np.array([-15,-15])
    lams += [lam]
    ts += [t]

    # Simple traj
    t = np.arange(0,Starts[1][0]+.01,.01)
    a = np.log(Starts[1][1]+1)/Starts[1][0]
    traj = np.vstack([t,np.exp(a*t)-1]).T
    trajs += [traj[::-1,:]]
    lam = np.array([-2,-2])
    lams += [lam]
    ts += [t]

    # Off-boundary traj
    t = np.arange(0,Starts[2][0]+.01,.01)
    a = np.log(Starts[2][1]+1)/Starts[2][0]
    traj = np.vstack([t,np.exp(a*t)-1]).T
    trajs += [traj[::-1,:]]
    lam = np.array([-2,-2])
    lams += [lam]
    ts += [t]

    groups = groupstuff(grid,trajs,lams,ts,Starts)
    p = procgroup(groups,grid)

    cmap = cm.jet
    norm = mpl.colors.Normalize(vmin=0.,vmax=1.)
    to_rgba = cm.ScalarMappable(norm=norm, cmap=cmap).to_rgba

    adjusted = np.zeros((np.sum(p < 1),2))
    adjcolor = np.zeros((np.sum(p < 1),4))
    shape = tuple(grid[:,2])
    i = 0
    for idp,pi in enumerate(p):
        if pi < 1:
            ind = int2ind(idp,shape)
            adjusted[i,:] = ind2pt2(ind,grid)
            adjcolor[i,:] = to_rgba(pi)
            i += 1

    plt.scatter(adjusted[:,0],adjusted[:,1],color=adjcolor)

    # Boundary
    x = np.arange(-1,3,.01)
    plt.plot(x,(x-1)**2+1.05,'k--',lw=2)

    for idt,traj in enumerate(trajs):
        lam = lams[idt]
        c = np.max(np.abs(lam))
        t = ts[idt]
        T = t[-1]
        z = [np.exp(-c*ti/T) for ti in t]
        colorline(traj[:,0],traj[:,1],z=z,cmap=cmap,norm=norm,linewidth=2)

    plt.scatter(trajs[0][-1][0],trajs[0][-1][1],marker='*',s=200,zorder=2,facecolors='w')
    plt.scatter(trajs[1][-1][0],trajs[1][-1][1],marker='*',s=200,zorder=2,facecolors='w')

    # plt.axis('equal')
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

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    Lambda_spi = ','.join('%.1f' % lam for lam in lams[0])
    Lambda_reg = ','.join('%.1f' % lam for lam in lams[1])
    textstr_spi = '$\Lambda=[$'+Lambda_spi+'$]$'
    textstr_reg = '$\Lambda=[$'+Lambda_reg+'$]$'
    ax.text(0.5, 0.95, textstr_spi, transform=ax.transAxes, fontsize=14,
            va='center', ha='center', bbox=props)
    ax.text(0.775, 0.1, textstr_reg, transform=ax.transAxes, fontsize=14,
            va='center', ha='center', bbox=props)

    ax.text(trajs[0][0,0]-.09, trajs[0][0,1]+.02, '$x_0$', fontsize=18,
            va='center', ha='center', fontweight='bold')
    ax.text(trajs[1][0,0]+.09, trajs[1][0,1]-.02, '$x_1$', fontsize=18,
            va='center', ha='center', fontweight='bold')
    ax.text(trajs[2][0,0]+.09, trajs[2][0,1], '$x_2$', fontsize=18,
            va='center', ha='center', fontweight='bold')

    ax.annotate('Boundary', xy=(.38, 1.38), xycoords='data',
                xytext=(.3, 1.), textcoords='data',
                arrowprops=dict(arrowstyle='simple',facecolor='black'),
                ha='center', va='center', size=18,
                )

    fig.suptitle('BoA Algorithm Enhanced with Heuristic',fontsize=18)
    plt.axes().set_aspect('equal')
    # plt.show()
    # plt.axis('off')
    plt.savefig("Heuristic.png",bbox_inches='tight')

    # embed()

if __name__ == '__main__':
    Demo()
