import numpy as np

step = 1e-5

# exact (grad x F) x F  for affine F=Ax+b given F from VI(F,K) and assuming algorithm is x = x - alpha F
def F_perp_exact(x,F,A):
    return np.dot(A-A.T,F(x,A))

# approximate (grad x F) x F direction
def F_perp_approx(x,F,A,alpha=step):
    return s(x,F,A,alpha)*F_perp_norm1(x,F,A,alpha)

# approximate (grad x F) x F direction, scale undetermined
def F_perp_norm1(x,F,A,alpha=step):
    Fx = F(x,A)
    Fxn = F(x-alpha*Fx,A)
    norm = np.linalg.norm(Fx)**2
    scalar = np.dot(Fx,Fxn)
    direc = -Fxn*norm + Fx*scalar
    return direc/np.linalg.norm(direc)

# correct scale for approximate direction
def s(x,F,A,alpha=step):
    norm = np.linalg.norm(F(x,A))
    return norm*curl_dot_normal(x,F,A,alpha)

# contour integral divided by area of triangle
def curl_dot_normal(x,F,A,alpha=step):
    Fx = F(x,A)
    x1 = x-alpha*Fx
    Fx1 = F(x1,A)
    x2 = x1-alpha*Fx1
    Fx2 = F(x2,A)
    w1 = 0.5*np.dot(Fx+Fx1,x1-x)
    w2 = 0.5*np.dot(Fx1+Fx2,x2-x1)
    w3 = 0.5*np.dot(Fx2+Fx,x-x2)
    A = herons(x,x1,x2)
    return -(w1+w2+w3)/A

# Jacobian-vector product with vector as F
def JF(x,F,A,alpha=step):
    Fx = F(x,A)
    x1 = x+alpha*Fx
    Fx1 = F(x1,A)
    return (Fx1-Fx)/alpha

# Jacobian-transpose F-vector product
def JTF(x,F,A,alpha=step):
    return JF(x,F,A,alpha) - F_perp_approx(x,F,A,alpha)

# herons rule for computing area
def herons(p1,p2,p3):
    a,b,c = np.linalg.norm(p2-p1), np.linalg.norm(p3-p2), np.linalg.norm(p1-p3)
    s = 0.5*(a+b+c)
    return np.sqrt(s*(s-a)*(s-b)*(s-c))

# affine field
def F(x,A,b=0.):
    return np.dot(A,x) + b

# random null vector (unit norm) for v*x = 0
def null(v):
    nv = np.random.rand(len(v))
    dims = np.random.permutation(len(v))
    dim = dims[np.nonzero(v[dims])[0][0]]
    s = np.dot(v,nv) - v[dim]*nv[dim]
    nv[dim] = -s/v[dim]
    return nv/np.linalg.norm(nv)



if __name__ == '__main__':
    dim = 3
    for i in range(10):
        x = np.random.rand(dim)
        # x = np.array([0,1])
        A = np.random.rand(dim,dim)
        # A = np.array([[0,-1],[1,0]])
        print(F_perp_approx(x,F,A))
        print(F_perp_exact(x,F,A))
        assert np.allclose(F_perp_approx(x,F,A),F_perp_exact(x,F,A),rtol=1e-2,atol=1e-3)
        assert np.allclose(JF(x,F,A),np.dot(A,F(x,A)),rtol=1e-2,atol=1e-3)
        assert np.allclose(JTF(x,F,A),np.dot(A.T,F(x,A)),rtol=1e-2,atol=1e-3)
