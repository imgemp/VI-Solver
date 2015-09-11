# import time
import numpy as np

from VISolver.Domains.CloudServices2 import (
    CloudServices, CreateNetworkExample)

# from VISolver.Solvers.Euler_LEGS import Euler_LEGS
from VISolver.Solvers.HeunEuler_LEGS import HeunEuler_LEGS
# from VISolver.Solvers.AdamsBashforthEuler_LEGS import ABEuler_LEGS
# from VISolver.Solvers.CashKarp_LEGS import CashKarp_LEGS

from VISolver.Projection import BoxProjection
from VISolver.Solver import Solve
from VISolver.Options import (
    DescentOptions, Miscellaneous, Reporting, Termination, Initialization)
from VISolver.Log import PrintSimStats

from VISolver.Utilities import (
    ListONP2NP, aug_grid, MCLET_BofA_ID_par, ind2int)

from matplotlib.colors import colorConverter
from matplotlib import pyplot as plt
import matplotlib as mpl

from IPython import embed

import time

from sklearn.svm import SVC


def Demo():

    #__CLOUD_SERVICES__##################################################

    # Define Network and Domain
    Network = CreateNetworkExample(ex=2)
    Domain = CloudServices(Network=Network,gap_alpha=2)

    # Set Method
    eps = 1e-2
    # Method = Euler_LEGS(Domain=Domain,P=BoxProjection(lo=eps))
    Method = HeunEuler_LEGS(Domain=Domain,P=BoxProjection(lo=eps),Delta0=1e0)
    # Method = ABEuler_LEGS(Domain=Domain,P=BoxProjection(lo=eps),Delta0=1e-1)
    # Method = CashKarp_LEGS(Domain=Domain,P=BoxProjection(lo=eps),Delta0=1e-6)

    # Set Options
    Init = Initialization(Step=-1e-3)
    Term = Termination(MaxIter=1e5)  # ,Tols=[(Domain.valid,False)])
    Repo = Reporting(Requests=['Data','Step'])
    Misc = Miscellaneous()
    Options = DescentOptions(Init,Term,Repo,Misc)
    args = (Method,Domain,Options)
    sim = Solve

    # Print Stats
    PrintSimStats(Domain,Method,Options)

    grid = [np.array([.5,3.5,6])]*Domain.Dim
    grid = ListONP2NP(grid)
    grid = aug_grid(grid)
    Dinv = np.diag(1./grid[:,3])

    results = MCLET_BofA_ID_par(sim,args,grid,nodes=16,limit=40,AVG=0.00,
                                eta_1=1.2,eta_2=.95,eps=1.,
                                L=16,q=8,r=1.1,Dinv=Dinv)
    ref, data, p, iters, avg, bndry_ids, starts = results

    sim_data = [results,Domain,grid]
    np.save('cloud_'+time.strftime("%Y%m%d_%H%M%S"),sim_data)

# HATCH VERSION

# Observed dimensions + fixed values for others
obs = (4,9)  # Look at new green-tech company
# consts = np.array([np.inf,np.inf,4.5,0.5,2.9,1.3,3.7,2.9,3.7,4.5])
consts = 2.9*np.ones(Domain.Dim)

# grid test cases for learned SVM classifier
xx, yy = np.meshgrid(np.linspace(grid[obs[0],0],grid[obs[0],1],500),
                     np.linspace(grid[obs[1],0],grid[obs[1],1],500))
padding = np.ones(len(xx.ravel()))
test = ()
for i in xrange(Domain.Dim):
    if i == obs[0]:
        test += (xx.ravel(),)
    elif i == obs[1]:
        test += (yy.ravel(),)
    else:
        test += (padding*consts[i],)
test = np.vstack(test).T

plt.figure()
ax = plt.gca()
c = plt.cm.hsv(np.random.rand(len(ref)))
white = colorConverter.to_rgba('white')
hatches = ('-', '+', 'x', '\\', '*', 'o', 'O', '.', '/')
Zs = np.zeros((500,500,len(ref)))
dist_max = np.linalg.norm(grid[:,1]-grid[:,0])
mydict = {}
for cat,lam in enumerate(ref):

    samples = data[hash(str(lam))]
    if samples != []:
        n = len(samples)
        m = len(samples[0][0])
        X = np.empty((n*2,m))
        for idx,sample in enumerate(samples):
            X[idx] = sample[0]
            X[idx+len(samples)] = sample[1]

            diff = sample[0]-consts
            diff[obs[0]] = 0
            diff[obs[1]] = 0
            dist = np.linalg.norm(diff)

            key = (sample[0][obs[0]],sample[0][obs[1]])
            if not key in mydict:
                mydict[key] = (dist,cat)
            elif dist < mydict[key][0]:
                mydict[key] = (dist,cat)

        Y = np.zeros(len(samples)*2)
        Y[:n] = 1

        clf = SVC()
        clf.fit(X,Y)

        Z = clf.decision_function(test)
        Z = Z.reshape(xx.shape)
        Zs[:,:,cat] = Z

txt_locs = [(1.4798387096774193, 0.82031250000000011),
            (3.1189516129032255, 1.046875),
            (1.625, 2.328125),
            (3.258064516129032, 2.921875)]

best_guess = np.argmax(Zs,axis=2)
cat_num = 0
for cat in set(best_guess.flatten()):
    Zma = np.ma.masked_where(best_guess != cat,Zs[:,:,cat])
    # cmap = mpl.colors.LinearSegmentedColormap.from_list('my_cmap',
    #                                                     [white,c[cat]],256)
    cmap = plt.cm.Greys
    cmap.set_bad(color='w',alpha=0.0)

    # for decision boundaries
    Zs_notcat = np.concatenate((Zs[:,:,:cat],Zs[:,:,cat+1:]),axis=2)
    diff = Zs[:,:,cat] - np.max(Zs_notcat,axis=2)

    # idx = np.unravel_index(np.argmax(diff),diff.shape)
    idx = np.unravel_index(np.argmax(Zma),Zma.shape)
    x = grid[obs[0],0] + (grid[obs[0],1]-grid[obs[0],0])/499*idx[1]
    y = grid[obs[1],0] + (grid[obs[1],1]-grid[obs[1],0])/499*idx[0]

    lam = ref[cat]
    if max(lam) < 0:
        dyn = 'stable'
    else:
        dyn = 'unstable'
    # print(x,y)
    # for placing text, see link below
    # http://stackoverflow.com/questions/25521120/
    # store-mouse-click-event-coordinates-with-matplotlib
    x,y = txt_locs[cat_num]
    plt.text(x,y,dyn,fontsize=12,ha='center',va='center',zorder=3,weight='bold',
             color='black',
             bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'))

    plt.contour(xx, yy, diff, colors='k', levels=[0], linewidths=2,
                linetypes='.-', zorder=1)

    mult, hat_num = divmod(cat_num,len(hatches))
    hatch = (mult+1)*hatches[hat_num]
    plt.contourf(xx,yy,Zma,0,hatches=[hatch],colors='none')

    plt.imshow(Zma, interpolation='nearest',
               extent=(xx.min(), xx.max(), yy.min(), yy.max()),
               aspect='auto', origin='lower', cmap=cmap, zorder=0)
    cat_num += 1
    # plt.draw()
    # plt.pause(2.0)

if obs[0] >= Domain.Dim // 2:
    xlabel = '$d_{' + str(obs[0]-Domain.Dim//2) + '}$'
else:
    xlabel = '$p_{' + str(obs[0]) + '}$'
if obs[1] >= Domain.Dim // 2:
    ylabel = '$d_{' + str(obs[1]-Domain.Dim//2) + '}$'
else:
    ylabel = '$p_{' + str(obs[1]) + '}$'
plt.xlabel(xlabel,fontsize=16)
plt.ylabel(ylabel,fontsize=16)

ax.set_xlim([grid[obs[0],0],grid[obs[0],1]])
ax.set_ylim([grid[obs[1],0],grid[obs[1],1]])
ax.set_aspect('equal')
plt.title('Boundaries of Attraction for Cloud Services Market',fontsize=18)

coords = []
fig = plt.gcf()
def onclick(event):
    global ix, iy
    ix, iy = event.xdata, event.ydata
    print 'x = %d, y = %d'%(
        ix, iy)

    global coords
    coords.append((ix, iy))

    # if len(coords) == 2:
    #     fig.canvas.mpl_disconnect(cid)

    return coords
cid = fig.canvas.mpl_connect('button_press_event', onclick)

fig.canvas.mpl_disconnect(cid)

plt.show()
plt.savefig('BoA.png',bbox_inches='tight')


# COLOR VERSION

# # Observed dimensions + fixed values for others
# obs = (4,9)  # Look at new green-tech company
# # consts = np.array([np.inf,np.inf,4.5,0.5,2.9,1.3,3.7,2.9,3.7,4.5])
# consts = 2.9*np.ones(Domain.Dim)

# # grid test cases for learned SVM classifier
# xx, yy = np.meshgrid(np.linspace(grid[obs[0],0],grid[obs[0],1],500),
#                      np.linspace(grid[obs[1],0],grid[obs[1],1],500))
# padding = np.ones(len(xx.ravel()))
# test = ()
# for i in xrange(Domain.Dim):
#     if i == obs[0]:
#         test += (xx.ravel(),)
#     elif i == obs[1]:
#         test += (yy.ravel(),)
#     else:
#         test += (padding*consts[i],)
# test = np.vstack(test).T

# plt.figure()
# ax = plt.gca()
# c = plt.cm.hsv(np.random.rand(len(ref)))
# white = colorConverter.to_rgba('white')
# Zs = np.zeros((500,500,len(ref)))
# dist_max = np.linalg.norm(grid[:,1]-grid[:,0])
# mydict = {}
# for cat,lam in enumerate(ref):

#     samples = data[hash(str(lam))]
#     if samples != []:
#         n = len(samples)
#         m = len(samples[0][0])
#         X = np.empty((n*2,m))
#         for idx,sample in enumerate(samples):
#             X[idx] = sample[0]
#             X[idx+len(samples)] = sample[1]

#             diff = sample[0]-consts
#             diff[obs[0]] = 0
#             diff[obs[1]] = 0
#             dist = np.linalg.norm(diff)

#             key = (sample[0][obs[0]],sample[0][obs[1]])
#             if not key in mydict:
#                 mydict[key] = (dist,cat)
#             elif dist < mydict[key][0]:
#                 mydict[key] = (dist,cat)

#         Y = np.zeros(len(samples)*2)
#         Y[:n] = 1

#         clf = SVC()
#         clf.fit(X,Y)

#         Z = clf.decision_function(test)
#         Z = Z.reshape(xx.shape)
#         Zs[:,:,cat] = Z

# # xscat = []
# # yscat = []
# # colors = []
# # sizes = []
# # lws = []
# # for key, value in mydict.iteritems():
# #     dist, cat = value
# #     color = c[cat]
# #     alpha = (dist_max - dist)/dist_max
# #     color[-1] = alpha

# #     xscat.append(key[0])
# #     yscat.append(key[1])
# #     colors.append(color)
# #     sizes.append(29*alpha+1)
# #     lws.append((alpha == 1.)*2)
# # plt.scatter(xscat,yscat,s=sizes,c=colors,lw=lws,zorder=2)

# best_guess = np.argmax(Zs,axis=2)
# for cat in set(best_guess.flatten()):
#     Zma = np.ma.masked_where(best_guess != cat,Zs[:,:,cat])
#     cmap = mpl.colors.LinearSegmentedColormap.from_list('my_cmap',
#                                                         [white,c[cat]],256)
#     cmap.set_bad(color='w',alpha=0.0)

#     # for decision boundaries
#     Zs_notcat = np.concatenate((Zs[:,:,:cat],Zs[:,:,cat+1:]),axis=2)
#     diff = Zs[:,:,cat] - np.max(Zs_notcat,axis=2)

#     # idx = np.unravel_index(np.argmax(diff),diff.shape)
#     idx = np.unravel_index(np.argmax(Zma),Zma.shape)
#     x = grid[obs[0],0] + (grid[obs[0],1]-grid[obs[0],0])/499*idx[1]
#     y = grid[obs[1],0] + (grid[obs[1],1]-grid[obs[1],0])/499*idx[0]

#     lam = ref[cat]
#     if max(lam) < 0:
#         dyn = 'stable'
#     else:
#         dyn = 'unstable'
#     print(x,y)
#     # for placing text, see link below
#     # http://stackoverflow.com/questions/25521120/
#     # store-mouse-click-event-coordinates-with-matplotlib
#     plt.text(x,y,dyn,fontsize=12,ha='center',va='center',zorder=3,weight='bold',
#              bbox=dict(facecolor='black', alpha=0.3, boxstyle='round'))

#     plt.contour(xx, yy, diff, colors='k', levels=[0], linewidths=2,
#                 linetypes='.-', zorder=1)
#     inner = .1*np.max(diff)
#     print(np.max(diff))
#     # if max(lam) < 0:
#     #     plt.contour(xx, yy, diff, colors='k', levels=[inner], linewidths=2,
#     #                 linestyles='--', zorder=1)
#     # else:
#     #     plt.contour(xx, yy, diff, colors='k', levels=[inner], linewidths=2,
#     #                 linestyles=':', zorder=1)

#     plt.imshow(Zma, interpolation='nearest',
#                extent=(xx.min(), xx.max(), yy.min(), yy.max()),
#                aspect='auto', origin='lower', cmap=cmap, zorder=0)
#     plt.draw()
#     plt.pause(0.5)

# ax.set_xlim([grid[obs[0],0],grid[obs[0],1]])
# ax.set_ylim([grid[obs[1],0],grid[obs[1],1]])
# ax.set_aspect('equal')
# plt.savefig('BoA.png',bbox_inches='tight')

# plt.figure()
# pmap = np.zeros((grid[obs[1],2],grid[obs[0],2]))
# ind_exp = np.array([int(i) for i in (consts-grid[:,0])//grid[:,3]])
# for ind_x in xrange(int(grid[obs[0],2])):
#     for ind_y in xrange(int(grid[obs[1],2])):
#         ind_exp[obs[0]] = ind_x
#         ind_exp[obs[1]] = ind_y
#         p_id = ind2int(tuple(ind_exp),tuple(grid[:,2]))
#         pmap[ind_y,ind_x] = p[p_id]
# plt.imshow(pmap,cmap='jet',origin='lower')
# plt.gca().set_aspect('equal')

# plt.show()

# embed()

if __name__ == '__main__':
    Demo()
