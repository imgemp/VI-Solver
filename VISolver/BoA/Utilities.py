from __future__ import division
import numpy as np
import random
import itertools


# Augment grid with grid increments or number per dimension
def aug_grid(grid,op=1):
    assert len(grid.shape) == 2
    assert all(grid[:,0] < grid[:,1])
    assert all(grid[:,2] > 0)
    if grid.shape[1] < 4:
        if op == 1:
            inc = (grid[:,1]-grid[:,0])/(grid[:,2]-1)
            return np.hstack((grid,inc[:,None]))
        else:
            N = (grid[:,1]-grid[:,0])/grid[:,2] + 1
            return np.hstack((grid[:,:2],N[:,None],grid[:,2]))
    else:
        assert all(grid[:,3] == (grid[:,1]-grid[:,0])/(grid[:,2]-1))
        print('Grid already augmented.')
        return grid


# Convert integer representation (ids) to grid indices
# Note: shape is tuple with number of points along each dimension of the grid
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


# Convert grid indices to cartesian points
# Note: grid is column array of start, end, N, inc
def ind2pt(ind,grid,checkBnds=False):
    assert len(ind) == grid.shape[0]
    if checkBnds:
        assert all(i < grid[j,2] for j,i in enumerate(ind))
        assert all(i >= 0 for i in ind)
    return grid[:,0] + np.multiply(ind,grid[:,3])


# Convert grid indices to integer representation (ids)
def ind2int(ind,shape):
    assert len(ind) == len(shape)
    assert all(x < y for x,y in zip(ind,shape))
    assert all(x >= 0 for x in ind)
    sizes = np.cumprod(shape[:0:-1])[::-1]
    return int(np.dot(ind[:-1],sizes)+ind[-1])


# Determine cube of grid points surrounding point pt
def pt2inds(pt,grid,checkBnds=False):
    lo = np.array([int(i) for i in (pt-grid[:,0])//grid[:,3]])
    rng = 2*np.ones(len(lo))
    if checkBnds:
        bnds = []
        for idx in np.ndindex(*rng):
            vert = tuple(np.add(idx,lo))
            if all(i >= 0 and i < grid[j,2] for j,i in enumerate(vert)):
                bnds.append(vert)
        return bnds
    else:
        return [tuple(np.add(idx,lo)) for idx in np.ndindex(*rng)]


# Determine all grid points within maximum distance, r, of grid index
# Note: Metric tensor Dinv not currently used - WIP
def neighbors(ind,grid,r,q=None,Dinv=1):
    lo = grid[:,0]
    hi = grid[:,1]
    inc = grid[:,3]

    # Initialize maximum possible neighbor grid
    i_max = np.array([int(v) for v in r//inc])
    cube = np.ndindex(*(i_max*2+1))

    # Grow neighbor grid by scanning cube
    neigh = []
    for idx in cube:
        offset = [v - i_max[k] for k,v in enumerate(idx)]
        if any(v != 0 for k,v in enumerate(offset)):  # if not origin
            # Convert index to cartesian
            n = np.add(offset,ind)
            loc = lo+n*inc
            # Add grid index point if in grid range and within r of index
            if all(loc >= lo) and all(loc <= hi) and \
               np.linalg.norm(offset*inc) < r:
                neigh += [tuple(n)]

    # Return all neighbors
    if q is None:
        return neigh
    # Return a few random neighbors
    selected = random.sample(neigh,min(q,len(neigh)))
    return selected, neigh


# Update LE reference with any new references
def update_LERef(ref,les,eps,data,ref_ept,endpts):
    # Initialize reference dictionaries if don't exist
    if ref is None:
        ref = les[0].copy()[None]
        ref_ept = endpts[0].copy()[None]
        data[hash(repr(les[0]))] = []

    # Check each LE against LE reference dictionary
    for l,le in enumerate(les):
        # Does LE have the same endpoint as any reference endpoints?
        ept = endpts[l]
        same_ept = [np.allclose(ept,_ref,rtol=.1,atol=1.) for _ref in ref_ept]

        # If there is an endpoint match, relax tolerance for LE match
        same = []
        for e,is_same_ept in enumerate(same_ept):
            if is_same_ept:
                same += [np.allclose(le,ref[e],rtol=.4,atol=1.)]
            else:
                same += [np.allclose(le,ref[e],rtol=.2,atol=1.)]

        # If LE doesn't match any in the reference dictionary, add LE to it
        if not any(same):
            ref = np.concatenate((ref,[le]))
            ref_ept = np.concatenate((ref_ept,[ept]))
            data[hash(repr(le))] = []
    return ref, data, ref_ept


# Map LEs to LEs in the reference dictionary
def adjustLEs2Ref(ref,les):
    # Map LE to closest LE in reference dictionary
    for idx,le in enumerate(les):
        diff = ref - le
        les[idx] = ref[np.argmin(np.linalg.norm(diff,axis=1))]


# Update probabilities of selected grid points and record boundary pairs
# Note: ids are list of ints representing sampled points with center at index 0
# Note: Reza's BoA pseudocode is unclear on updates, code is inconsistent
def update_Prob_Data(ids,shape,grid,les,eps,p,eta_1,eta_2,data):
    # Track grid points that have already been identified as boundary points
    toZero = set()

    # Scan all combinations of grid points (all grid boundary possibilities)
    boundry_pairs = 0
    for pair in itertools.combinations(np.arange(len(les)),2):
        le_a = les[pair[0]]
        le_b = les[pair[1]]
        same = all(le_a == le_b)
        if not same:
            # Increase probability of points if boundary pair found in group
            boundry_pairs += 1
            id_a = ids[pair[0]]
            id_b = ids[pair[1]]
            toZero.update([id_a,id_b])
            p[ids[1:]] *= eta_1
            # Add pair to corresponding dataset
            pt_a = ind2pt(int2ind(id_a,shape),grid)
            pt_b = ind2pt(int2ind(id_b,shape),grid)
            data[hash(repr(le_a))] += [[pt_a,pt_b]]
            data[hash(repr(le_b))] += [[pt_b,pt_a]]
        else:
            # Decrease probablitiy of pair
            p[ids[pair[0]]] *= eta_2
            p[ids[pair[1]]] *= eta_2
    # Prevent drawing the same grid point from p
    for z in toZero:
        p[z] = 0
    return p, data, boundry_pairs, toZero
