import numpy as np

from matplotlib.colors import colorConverter
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap as lsc

from VISolver.BoA.Utilities import ind2int

from sklearn.svm import SVC


def onclick(event):
    # Records click locations to global variable on click event
    if not 'coords' in globals():
        global coords
        coords = []
    ix, iy = event.xdata, event.ydata
    print('x = %g, y = %g' % (ix, iy))
    coords.append((ix, iy))
    return


def plotBoA(ref,data,grid,obs=None,consts=None,Neval=500,txt_locs=None,
            wcolor=False,wscatter=False,makeTxt_Locs=False,saveFig=False,
            xlabel=None,ylabel=None,title='Boundaries of Attraction'):

    if len(ref) <= 1:
        print('No Boundaries Exist!')
        return

    # Handle observable dimensions and fixed (hel constant) dimensions
    if obs is None:
        obs = np.arange(min(grid.shape[0],2))
    if consts is None:
        consts = grid[:,0]
    if isinstance(obs,int):
        obs = [obs]
    assert len(obs) <= 2

    # Construct test cases for SVM classifier
    if len(obs) == 2:
        xx, yy = np.meshgrid(np.linspace(grid[obs[0],0],grid[obs[0],1],Neval),
                             np.linspace(grid[obs[1],0],grid[obs[1],1],Neval))
        padding = np.ones(len(xx.ravel()))
        test = ()
        for i in range(grid.shape[0]):
            if i == obs[0]:
                test += (xx.ravel(),)
            elif i == obs[1]:
                test += (yy.ravel(),)
            else:
                test += (padding*consts[i],)
        test = np.vstack(test).T
    else:
        xx, yy = np.meshgrid(np.linspace(grid[obs[0],0],grid[obs[0],1],Neval),
                             np.linspace(0,1,2))
        test = np.tile(consts,(Neval,1))
        test[:,obs[0]] = np.linspace(grid[obs[0],0],grid[obs[0],1],Neval)

    # Define colormaps and hatches for plotting
    c = plt.cm.hsv(np.random.rand(len(ref)))
    white = colorConverter.to_rgba('white')
    dist_max = np.linalg.norm(grid[:,1]-grid[:,0])
    hatches = ('-', '\\', '*', 'o', '+', 'x', 'O', '.', '/')

    # Learn SVM Classifiers for each basin of attraction and
    # compute decision function over grid
    if len(obs) == 2:
        Zs = np.zeros((Neval,Neval,len(ref)))
    else:
        Zs = np.zeros((2,Neval,len(ref)))
    mydict = {}
    for cat,lam in enumerate(ref):

        samples = data[hash(repr(lam))]
        if samples != []:
            n = len(samples)
            m = len(samples[0][0])
            X = np.empty((n*2,m))
            for idx,sample in enumerate(samples):
                X[idx] = sample[0]
                X[idx+len(samples)] = sample[1]

                diff = sample[0]-consts
                for o in obs:
                    diff[o] = 0
                dist = np.linalg.norm(diff)

                key = tuple([sample[0][o] for o in obs])
                if not key in mydict:
                    mydict[key] = (dist,cat)
                elif dist < mydict[key][0]:
                    mydict[key] = (dist,cat)

            Y = np.zeros(len(samples)*2)
            Y[:n] = 1

            clf = SVC()
            clf.fit(X,Y)

            Z = clf.decision_function(test)
            if len(obs) == 2:
                Z = Z.reshape(xx.shape)
                Zs[:,:,cat] = Z
            else:
                Z = np.tile(Z.T,(2,1))
                Zs[:,:,cat] = Z

    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Plot data points from training set
    if wscatter:
        xscat = []
        if len(obs) == 2:
            yscat = []
        colors, sizes, lws = [], [], []
        for key, value in mydict.iteritems():
            dist, cat = value
            if wcolor:
                color = c[cat]
            else:
                color = [0,0,0,0]
            alpha = (dist_max - dist)/dist_max
            color[-1] = alpha

            xscat.append(key[0])
            if len(obs) == 2:
                yscat.append(key[1])
            colors.append(color)
            sizes.append(29*alpha+1)
            lws.append((alpha == 1.)*2.)

        if wcolor:
            if len(obs) == 2:
                ax.scatter(xscat,yscat,s=sizes,c=colors,lw=lws,zorder=2)
            else:
                for xid, xval in enumerate(xscat):
                    ax.plot([xval]*2,[0,1],c=colors[xid],lw=lws[xid],zorder=2)
        else:
            if len(obs) == 2:
                ax.scatter(xscat,yscat,s=sizes,c=colors,lw=lws,zorder=2)
            else:
                for xid, xval in enumerate(xscat):
                    ax.plot([xval]*2,[0,1],c=colors[xid],lw=lws[xid],zorder=2)

    # Plot decision boundaries and shade/hatch/categorize basins of attraction
    best_guess = np.argmax(Zs,axis=2)
    cat_num = 0
    for cat in set(best_guess.flatten()):
        Zma = np.ma.masked_where(best_guess != cat,Zs[:,:,cat])

        if wcolor:
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
            if len(obs) == 2:
                y = grid[obs[1],0] + (grid[obs[1],1]-grid[obs[1],0])/499*idx[0]
            else:
                y = 0.5
        else:
            x,y = txt_locs[cat_num]

        lam = ref[cat]
        if max(lam) < 0:
            dyn = 'stable'
        elif max(lam) == 0.:
            dyn = 'torus'
        else:
            dyn = 'unstable'

        ax.text(x,y,dyn,fontsize=12,ha='center',va='center',zorder=3,
                weight='bold',color='black',
                bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'))

        ax.contour(xx, yy, diff, colors='k', levels=[0], linewidths=2,
                   linetypes='.-', zorder=1)

        if not wcolor:
            mult, hat_num = divmod(cat_num,len(hatches))
            hatch = (mult+1)*hatches[hat_num]
            ax.contourf(xx,yy,Zma,0,hatches=[hatch],colors='none')

        ax.imshow(Zma, interpolation='nearest',
                  extent=(xx.min(), xx.max(), yy.min(), yy.max()),
                  aspect='auto', origin='lower', cmap=cmap, zorder=0)

        cat_num += 1

        if makeTxt_Locs:
            plt.draw()
            plt.pause(2.0)

    # Label axes
    if xlabel is None:
        xlabel = '$x_'+repr(obs[0])+'$'
    plt.xlabel(xlabel,fontsize=16)
    ax.set_xlim([grid[obs[0],0],grid[obs[0],1]])

    if len(obs) == 2:
        if ylabel is None:
            ylabel = '$x_'+repr(obs[1])+'$'
        plt.ylabel(ylabel,fontsize=16)
        ax.set_ylim([grid[obs[1],0],grid[obs[1],1]])
    else:
        ax.set_ylim([0,1])
        plt.tick_params(axis='y',which='both',bottom='off',top='off',
                        labelbottom='off')
        fig.set_size_inches(8, 3)

    # Title and 'square' axes
    ax.set_aspect('equal')
    fig.suptitle(title,fontsize=18)

    # Setup event handler for recording click locations
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

    return fig, ax


def plotDistribution(p,grid,obs=None,consts=None,saveFig=False):
    # Handle observable dimensions and fixed (hel constant) dimensions
    if obs is None:
        obs = np.arange(min(grid.shape[0],2))
    if consts is None:
        consts = grid[:,0]
    if isinstance(obs,int):
        obs = [obs]
    assert len(obs) == 2

    # Extract distribution map
    if grid.shape[0] == 2:
        pmap = np.swapaxes(np.reshape(p,tuple(grid[:,2])),0,1)
    else:
        pmap = np.zeros((grid[obs[1],2],grid[obs[0],2]))
        ind_exp = np.array([int(i) for i in (consts-grid[:,0])//grid[:,3]])
        for ind_x in range(int(grid[obs[0],2])):
            for ind_y in range(int(grid[obs[1],2])):
                ind_exp[obs[0]] = ind_x
                ind_exp[obs[1]] = ind_y
                p_id = ind2int(tuple(ind_exp),tuple(grid[:,2]))
                pmap[ind_y,ind_x] = p[p_id]

    # Plot distribution map
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(pmap,cmap='jet',origin='lower')
    ax.set_aspect('equal')

    if saveFig:
        plt.savefig('BoA.png',bbox_inches='tight')

    return fig, ax
