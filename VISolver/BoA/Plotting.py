import numpy as np

from matplotlib.colors import colorConverter
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap as lsc

from sklearn.svm import SVC


def onclick(event):
    if not 'coords' in globals():
        global coords
        coords = []
    ix, iy = event.xdata, event.ydata
    print 'x = %g, y = %g' % (ix, iy)
    coords.append((ix, iy))
    return


def plotBoA(ref,data,grid,obs=None,consts=None,Neval=500,txt_locs=None,
            color=False,scatter=False,makeTxt_Locs=False,saveFig=False):

    if obs is None:
        obs = np.arange(min(grid.shape[0],2))
    if consts is None:
        consts = grid[:,0]

    # Grid test cases for learned SVM classifier
    xx, yy = np.meshgrid(np.linspace(grid[obs[0],0],grid[obs[0],1],Neval),
                         np.linspace(grid[obs[1],0],grid[obs[1],1],Neval))
    padding = np.ones(len(xx.ravel()))
    test = ()
    for i in xrange(grid.shape[0]):
        if i == obs[0]:
            test += (xx.ravel(),)
        elif i == obs[1]:
            test += (yy.ravel(),)
        else:
            test += (padding*consts[i],)
    test = np.vstack(test).T

    fig = plt.figure()
    ax = fig.add_subplot(111)

    c = plt.cm.hsv(np.random.rand(len(ref)))
    white = colorConverter.to_rgba('white')
    hatches = ('-', '+', 'x', '\\', '*', 'o', 'O', '.', '/')

    Zs = np.zeros((Neval,Neval,len(ref)))
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

    best_guess = np.argmax(Zs,axis=2)
    cat_num = 0
    for cat in set(best_guess.flatten()):
        Zma = np.ma.masked_where(best_guess != cat,Zs[:,:,cat])

        if color:
            cmap = lsc.from_list('my_cmap',[white,c[cat]],256)
        else:
            cmap = plt.cm.Greys
        cmap.set_bad(color='w',alpha=0.0)

        # for decision boundaries
        Zs_notcat = np.concatenate((Zs[:,:,:cat],Zs[:,:,cat+1:]),axis=2)
        diff = Zs[:,:,cat] - np.max(Zs_notcat,axis=2)

        if txt_locs is None:
            idx = np.unravel_index(np.argmax(Zma),Zma.shape)
            x = grid[obs[0],0] + (grid[obs[0],1]-grid[obs[0],0])/499*idx[1]
            y = grid[obs[1],0] + (grid[obs[1],1]-grid[obs[1],0])/499*idx[0]
        else:
            x,y = txt_locs[cat_num]

        lam = ref[cat]
        if max(lam) < 0:
            dyn = 'stable'
        elif max(lam) == 0.:
            dyn = 'torus'
        else:
            dyn = 'unstable'

        plt.text(x,y,dyn,fontsize=12,ha='center',va='center',zorder=3,
                 weight='bold',color='black',
                 bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'))

        plt.contour(xx, yy, diff, colors='k', levels=[0], linewidths=2,
                    linetypes='.-', zorder=1)

        if color:
            plt.contour(xx, yy, Z, colors='k', levels=[0], linewidths=2,
                        linetypes='-.', zorder=1)
            if scatter:
                plt.scatter(X[:n, 0], X[:n, 1], s=30, c=c[cat], zorder=2)
        else:
            mult, hat_num = divmod(cat_num,len(hatches))
            hatch = (mult+1)*hatches[hat_num]
            plt.contourf(xx,yy,Zma,0,hatches=[hatch],colors='none')
            if scatter:
                plt.scatter(X[:n, 0], X[:n, 1], s=30, c='k', zorder=2)

        plt.imshow(Zma, interpolation='nearest',
                   extent=(xx.min(), xx.max(), yy.min(), yy.max()),
                   aspect='auto', origin='lower', cmap=cmap, zorder=0)
        cat_num += 1

        if makeTxt_Locs:
            plt.draw()
            plt.pause(2.0)

    plt.xlabel(str(obs[0]),fontsize=16)
    plt.ylabel(str(obs[1]),fontsize=16)

    ax.set_xlim([grid[obs[0],0],grid[obs[0],1]])
    ax.set_ylim([grid[obs[1],0],grid[obs[1],1]])
    ax.set_aspect('equal')
    fig.suptitle('Boundaries of Attraction',fontsize=18)

    if makeTxt_Locs:
        global coords
        coords = []
        cid = fig.canvas.mpl_connect('button_press_event', onclick)

    if makeTxt_Locs or not saveFig:
        plt.show()

    if makeTxt_Locs:
        fig.canvas.mpl_disconnect(cid)

    if saveFig:
        plt.savefig('BoA.png',bbox_inches='tight')

    plt.clf()
    return coords
