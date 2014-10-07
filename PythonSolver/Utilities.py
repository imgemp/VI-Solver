import numpy as np

#Utilities
def MachineLimit_Exp(Var,Const,L=-700.0,H=700.0):
    Var_mn = np.abs(Var)
    Var_mx = np.abs(Var)
    Const_mn = np.min(np.sign(Var)*Const)
    Const_mx = np.max(np.sign(Var)*Const)
    if np.abs(Var)*Const_mn < L: Var_mn = np.abs(L/Const_mn)
    if np.abs(Var)*Const_mx > H: Var_mx = np.abs(H/Const_mx)
    return np.min([Var_mn,Var_mx,np.abs(Var)])*np.sign(Var)