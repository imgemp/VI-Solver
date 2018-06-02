from __future__ import division
import numpy as np


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

    if U.shape[1] > 1:
        comp = U.dtype == np.complex
        for i in range(U.shape[1]):
            vi = A[:,i]
            proj = 0*vi
            for j in range(i):
                uj = U[:,j]
                proj += dot(vi,uj,comp)/dot(uj,uj,comp)*uj
            U[:,i] = vi - proj

    if normalize:
        return U/np.linalg.norm(U,axis=0)
    return U


def dot(a,b,comp=False):
    if comp:
        return np.dot(a,np.conj(b))
    else:
        return np.dot(a,b)


def Jv(Data,Psi,Jac,F_Data=None):
    dim = Data.size
    Psi_rsh = Psi.reshape((dim,-1))
    return np.dot(Jac(Data),Psi_rsh)


def Jv_num(Data,Psi,F,F_Data=None,eps=1e-10):
    Psi_rsh = np.atleast_2d(Psi.T).T
    if F_Data is None:
        F_Data = F(Data)
    res = np.zeros_like(Psi_rsh)
    for i,Psi_i in enumerate(Psi_rsh.T):
        res[:,i] = (F(Data+eps*Psi_i) - F_Data)/eps
    return res


def approx_jacobian(F,x,epsilon=np.sqrt(np.finfo(float).eps),*args):
    """Approximate the Jacobian matrix of callable function func
       * Rob Falck's implementation as a part of scipy.optimize.fmin_slsqp
       * Parameters
         x       - The state vector at which the Jacobian matrix is
         desired
         func    - A vector-valued function of the form f(x,*args)
         epsilon - The peturbation used to determine the partial derivatives
         *args   - Additional arguments passed to func

       * Returns
         An array of dimensions (lenf, lenx) where lenf is the length
         of the outputs of func, and lenx is the number of states

       * Notes
         The approximation is done using forward differences

    """
    func = F
    x0 = np.asfarray(x)
    f0 = func(*((x0,)+args))
    jac = np.zeros([len(x0),len(f0)])
    dx = np.zeros(len(x0))
    for i in range(len(x0)):
        dx[i] = epsilon
        jac[i] = (func(*((x0+dx,)+args)) - f0)/epsilon
        dx[i] = 0.0
    return jac.transpose()


def ListONP2NP(L):
    arr = np.empty((len(L),)+L[0].shape)
    for idx,x in enumerate(L):
        arr[idx] = x
    return arr


def UnpackFlattened(Data,shapes):
    assert isinstance(shapes,list)
    assert np.all([isinstance(shape,tuple) for shape in shapes])
    lengths = [np.prod(shape) for shape in shapes]
    cumlengths = np.cumsum([0]+[np.prod(shape) for shape in shapes])
    assert Data.shape == (cumlengths[-1],)

    return [Data[cumlengths[i]:cumlengths[i+1]].reshape(shapes[i]) for i in range(len(cumlengths)-1)]

def RandUnit(Data):
    v = np.random.rand(*Data.shape)
    return v/np.linalg.norm(v)
