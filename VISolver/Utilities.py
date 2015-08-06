import numpy as np
import random
import itertools


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


# grid contains list of (start,end,N) tuples of floats
def ind2pt(ind,grid):
    assert len(ind) == len(grid)
    pt = np.zeros(len(ind))
    for i in xrange(len(pt)):
        start, end, N = grid[i]
        assert ind[i] < N
        pt[i] = start+ind[i]*(end-start)/(N-1)
    return pt


def ind2int(ind,shape):
    assert len(ind) == len(shape)
    less = [(x < y) for x, y in zip(ind,shape)]
    assert all(less)
    more = [(x > 0) for x in ind]
    assert all(more)
    sizes = np.cumprod(shape[:0:-1])[::-1]
    i = np.dot(ind[:-1],sizes)+ind[-1]
    return i


def neighbors(ind,grid,r,q=None):
    inc = np.zeros(len(grid))
    for idx, tup in enumerate(grid):
        start, end, N = tup
        inc[idx] = (end-start)/(N-1)
    i_max = r//inc+1
    neigh = []
    cube = np.ndindex(*i_max)
    next(cube)  # skip origin
    for idx in cube:
        if np.linalg.norm(idx*inc) < r:
            neigh += [tuple(np.add(idx,ind))]
    if q is None:
        return neigh
    selected = random.sample(neigh,q)
    return selected, neigh


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return itertools.izip(a, b)


def update_LamRef(ref,lams,eps,data):
    for lam in lams:
        diff = ref - lam
        if not any(np.linalg.norm(diff,axis=1) <= eps):
            ref = np.concatenate((ref,[lam]))
            data[ref] = []
    return ref, data


def adjustLams2Ref(ref,lams):
    for idl,lam in enumerate(lams):
        diff = ref - lam
        lams[idl] = ref[np.argmin(np.linalg.norm(diff,axis=1))]
    return lams


# ids should be list of ints representing sampled points with center at index 0
def update_Prob_Data(ids,shape,grid,lams,eps,p,eta_1,eta_2,data):
    toZero = set()
    for pair in pairwise(np.arange(lams.shape[0])):
        lam_a = lams[pair[0]]
        lam_b = lams[pair[1]]
        same = np.linalg.norm(lam_a - lam_b) <= eps
        if not same:
            id_a = ids[pair[0]]
            id_b = ids[pair[1]]
            toZero.update([id_a,id_b])
            p[ids[1:]] += eta_1
            # add pair to corresponding dataset
            pt_a = ind2pt(int2ind(id_a,shape),grid)
            pt_b = ind2pt(int2ind(id_b,shape),grid)
            data[lam_a] += [pt_a,pt_b]
            data[lam_b] += [pt_a,pt_b]
        else:
            p[ids[pair[0]]] -= eta_2
            p[ids[pair[1]]] -= eta_2
    for z in toZero:
        p[z] = 0
    return p, data

# p = [1./np.prod(shape)]*np.prod(shape)
# ids = np.arange(np.prod(shape))
# iter = 0
# avg = np.inf
# L = 10
# grid = ?
# while (iter <= limit) and (avg > AVG):
#     centers = np.random.randint(ids,size=L,p=p)
#     inds = [int2ind(center,shape) for center in centers]
#     lams = []
#     for ind in inds:
#         selected, neigh = neighbors(ind,grid,r,q=None)
#         chosen = [ind2pt(item,grid) for item in selected + ind]
#         for start in chosen:
#             # run sim / compute lam
#             lams += lam
