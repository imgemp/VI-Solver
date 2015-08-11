import numpy as np
import random
import itertools
import pathos.multiprocessing as mp


#Utilities
def MachineLimit_Exp(Var, Const, L=-700., H=700.):
    Var_mn = np.abs(Var)
    Var_mx = np.abs(Var)
    Const_mn = np.min(np.sign(Var)*Const)
    Const_mx = np.max(np.sign(Var)*Const)
    if np.abs(Var)*Const_mn < L:
        Var_mn = np.abs(L/Const_mn)
    if np.abs(Var)*Const_mx > H:
        Var_mx = np.abs(H/Const_mx)
    return np.min([Var_mn,Var_mx,np.abs(Var)])*np.sign(Var)


def GramSchmidt(A,normalize=True):
    U = A.copy()
    for i in xrange(A.shape[0]):
        vi = A[:,i]
        proj = 0*vi
        for j in xrange(i):
            uj = U[:,j]
            proj += np.dot(vi,uj)/np.dot(uj,uj)*uj
        U[:,i] = vi - proj

    if normalize:
        return U/np.linalg.norm(U,axis=0)
    return U


def ListONP2NP(L):
    arr = np.empty((len(L),)+L[0].shape)
    for idx,x in enumerate(L):
        arr[idx] = x
    return arr


# Following functions added for grid sampling
def aug_grid(grid,op=1):
    if op == 1:
        inc = (grid[:,1]-grid[:,0])/(grid[:,2]-1)
        return np.hstack((grid,inc[:,None]))
    else:
        N = (grid[:,1]-grid[:,0])/grid[:,2] + 1
        return np.hstack((grid[:,:2],N[:,None],grid[:,2]))


# shape is tuple containing number of points along each dimension of the grid
def int2ind(i,shape):
    assert i >= 0
    assert i < np.prod(shape)
    ind = ()
    divisors = np.cumprod(shape[:0:-1])[::-1]
    for d in divisors:
        q,i = divmod(i,d)
        ind += (q,)
    ind += (i,)
    return ind


# grid is column array of start, end, N, inc
def ind2pt(ind,grid):
    assert len(ind) == grid.shape[0]
    assert all(i < grid[j,2] for j,i in enumerate(ind))
    assert all(i >= 0 for i in ind)
    return grid[:,0] + np.multiply(ind,grid[:,3])


def ind2int(ind,shape):
    assert len(ind) == len(shape)
    assert all(x < y for x,y in zip(ind,shape))
    assert all(x >= 0 for x in ind)
    sizes = np.cumprod(shape[:0:-1])[::-1]
    return int(np.dot(ind[:-1],sizes)+ind[-1])


def neighbors(ind,grid,r,q=None,Dinv=1):
    lo = grid[:,0]
    hi = grid[:,1]
    inc = grid[:,3]
    i_max = np.array([int(v) for v in r//inc])
    neigh = []
    cube = np.ndindex(*(i_max*2+1))
    for idx in cube:
        offset = [v - i_max[k] for k,v in enumerate(idx)]
        if any(v != 0 for k,v in enumerate(offset)):  # not origin
            n = np.add(offset,ind)
            loc = lo+n*inc
            if all(loc >= lo) and all(loc <= hi) and \
               np.linalg.norm(np.dot(offset*inc,Dinv)) < r:
                neigh += [tuple(n)]
    if q is None:
        return neigh
    selected = random.sample(neigh,min(q,len(neigh)))
    return selected, neigh


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return itertools.izip(a, b)


def update_LamRef(ref,lams,eps,data):
    if ref is None:
        ref = lams[0].copy()[None]
        data[hash(str(lams[0]))] = []
    for lam in lams:
        diff = ref - lam
        if all(np.linalg.norm(diff,axis=1) > eps*diff.shape[1]):
            ref = np.concatenate((ref,[lam]))
            data[hash(str(lam))] = []
    return ref, data


def adjustLams2Ref(ref,lams):
    for idx,lam in enumerate(lams):
        diff = ref - lam
        lams[idx] = ref[np.argmin(np.linalg.norm(diff,axis=1))]


# ids should be list of ints representing sampled points with center at index 0
def update_Prob_Data(ids,shape,grid,lams,eps,p,eta_1,eta_2,data):
    toZero = set()
    boundry_pairs = 0
    for pair in pairwise(np.arange(len(lams))):
        lam_a = lams[pair[0]]
        lam_b = lams[pair[1]]
        same = np.linalg.norm(lam_a - lam_b) <= eps
        if not same:
            boundry_pairs += 1
            id_a = ids[pair[0]]
            id_b = ids[pair[1]]
            toZero.update([id_a,id_b])
            p[ids[1:]] *= eta_1
            # add pair to corresponding dataset
            pt_a = ind2pt(int2ind(id_a,shape),grid)
            pt_b = ind2pt(int2ind(id_b,shape),grid)
            data[hash(str(lam_a))] += [[pt_a,pt_b]]
            data[hash(str(lam_b))] += [[pt_b,pt_a]]
        else:
            p[ids[pair[0]]] *= eta_2
            p[ids[pair[1]]] *= eta_2
    for z in toZero:
        p[z] = 0
    return p, data, boundry_pairs


def MCLE_BofA_Identification(sim,args,grid,limit=1,AVG=.01,eta_1=1.2,eta_2=.95,
                             eps=1.,L=1,q=2,r=1.1,Dinv=1):
    shape = tuple(grid[:,2])
    p = np.ones(np.prod(shape))/np.prod(shape)
    ids = range(int(np.prod(shape)))

    ref = None
    data = {}
    B_pairs = 0

    i = 0
    avg = np.inf
    while (i <= limit) or (avg > AVG):
        print(i)
        center_ids = np.random.choice(ids,size=L,p=p)
        center_inds = [int2ind(center_id,shape) for center_id in center_ids]
        groups = []
        for center_ind in center_inds:
            selected, neigh = neighbors(center_ind,grid,r,q,Dinv)
            group_inds = [center_ind] + selected
            group_ids = [ind2int(ind,shape) for ind in group_inds]
            group_pts = [ind2pt(ind,grid) for ind in group_inds]
            print(group_pts)
            lams = []
            for start in group_pts:
                results = sim(start,*args)
                lams += [results.TempStorage['Lyapunov'][-1]]
            ref, data = update_LamRef(ref,lams,eps,data)
            groups += [[group_ids,lams]]
        for group in groups:
            lams = group[1]
            adjustLams2Ref(ref,lams)
        for group in groups:
            group_ids, group_lams = group
            p, data, b_pairs = update_Prob_Data(group_ids,shape,grid,
                                                group_lams,eps,
                                                p,eta_1,eta_2,
                                                data)
            B_pairs += b_pairs
        p = p/np.sum(p)
        i += 1
        avg = B_pairs/((q+1)*L*i)
    return ref, data, p, i, avg


def compLEs(x):
    center_ind,sim,args,grid,shape,eps,q,r,Dinv = x
    selected, neigh = neighbors(center_ind,grid,r,q,Dinv)
    group_inds = [center_ind] + selected
    group_ids = [ind2int(ind,shape) for ind in group_inds]
    group_pts = [ind2pt(ind,grid) for ind in group_inds]
    lams = []
    for start in group_pts:
        results = sim(start,*args)
        lams += [results.TempStorage['Lyapunov'][-1]]
    return [group_ids,lams]


def MCLE_BofA_ID_par(sim,args,grid,nodes=8,limit=1,AVG=.01,eta_1=1.2,eta_2=.95,
                     eps=1.,L=1,q=2,r=1.1,Dinv=1):
    shape = tuple(grid[:,2])
    p = np.ones(np.prod(shape))/np.prod(shape)
    ids = range(int(np.prod(shape)))

    ref = None
    data = {}
    B_pairs = 0

    pool = mp.ProcessingPool(nodes=nodes)

    i = 0
    avg = np.inf
    while (i < limit) or (avg > AVG):
        print(i)
        center_ids = np.random.choice(ids,size=L,p=p)
        center_inds = [int2ind(center_id,shape) for center_id in center_ids]
        x = [(ind,sim,args,grid,shape,eps,q,r,Dinv) for ind in center_inds]
        groups = pool.map(compLEs,x)
        for group in groups:
            lams = group[1]
            ref, data = update_LamRef(ref,lams,eps,data)
        for group in groups:
            lams = group[1]
            adjustLams2Ref(ref,lams)
        for group in groups:
            group_ids, group_lams = group
            p, data, b_pairs = update_Prob_Data(group_ids,shape,grid,
                                                group_lams,eps,
                                                p,eta_1,eta_2,
                                                data)
            B_pairs += b_pairs
        p = p/np.sum(p)
        i += 1
        avg = B_pairs/((q+1)*L*i)
    return ref, data, p, i, avg
