import numpy as np

from VISolver.Projection import IdentityProjection
from VISolver.Solver import Solver
from VISolver.Utilities import GramSchmidt


class CashKarp_LEGS(Solver):

    def __init__(self, Domain, P=IdentityProjection(), Delta0=1e-4,
                 GrowthLimit=2, MinStep=-1e10, MaxStep=1e10):

        self.F = Domain.F

        self.Jac = Domain.Jac

        self.Proj = P

        self.StorageSize = 1

        self.TempStorage = {}

        self.Delta0 = Delta0

        self.GrowthLimit = GrowthLimit

        self.MinStep = MinStep

        self.MaxStep = MaxStep

        self.BT = np.array([
            [1./5.,0.,0.,0.,0.,0.],
            [3./40.,9./40.,0.,0.,0.,0.],
            [3./10.,-9./10.,6./5.,0.,0.,0.],
            [-11./54.,5./2.,-70./27.,35./27.,0.,0.],
            [1631./55296.,175./512.,575./13824.,44275./110592.,253./4096.,0.],
            [37./378.,0.,250./621.,125./594.,0.,512./1771.],
            [2825./27648.,0.,18575./48384.,13525./55296.,277./14336.,0.25]
        ])

    def InitTempStorage(self,Start,Domain,Options):

        self.TempStorage['Data'] = self.StorageSize*[Start]
        self.TempStorage[self.F] = self.StorageSize*[self.F(Start)]

        Psi_0 = np.eye(Start.size)
        dPsi_0 = np.dot(self.Jac(Start),Psi_0)
        self.TempStorage['Psi'] = self.StorageSize*[Psi_0.flatten()]
        self.TempStorage['dPsi'] = self.StorageSize*[dPsi_0.flatten()]
        self.TempStorage['Lyapunov'] = self.StorageSize*[0*Start]
        self.TempStorage['T'] = self.StorageSize*[0]

        self.TempStorage['Step'] = self.StorageSize*[Options.Init.Step]
        self.TempStorage['F Evaluations'] = self.StorageSize*[1]
        self.TempStorage['Projections'] = self.StorageSize*[0]

        return self.TempStorage

    # BookKeeping(self,TempData) defined in super class 'Solver'

    def Update(self,Record):

        # Retrieve Necessary Data
        Data_x = Record.TempStorage['Data'][-1]
        Data_psi = Record.TempStorage['Psi'][-1]
        Lyapunov = Record.TempStorage['Lyapunov'][-1]
        T = Record.TempStorage['T'][-1]
        dim = Data_x.size

        Fs_x = np.zeros((6,Data_x.shape[0]))
        Fs_x[0,:] = Record.TempStorage[self.F][-1]

        Fs_psi = np.zeros((6,Data_psi.shape[0]))
        Fs_psi[0,:] = Record.TempStorage['dPsi'][-1]

        Step = Record.TempStorage['Step'][-1]

        # Initialize Storage
        TempData = {}

        # Calculate k values (gradients)
        for i in xrange(5):
            direction_x = np.einsum('i,i...', self.BT[i,:i+1], Fs_x[:i+1])
            direction_psi = np.einsum('i,i...', self.BT[i,:i+1], Fs_psi[:i+1])

            _NewData_x = self.Proj.P(Data_x, Step, direction_x)
            _NewData_psi = (Data_psi+Step*direction_psi).reshape((dim,-1))

            Fs_x[i+1,:] = self.F(_NewData_x)
            Fs_psi[i+1,:] = np.dot(self.Jac(_NewData_x),_NewData_psi).flatten()

        # Compute order-p, order-p+1 data points
        direction_x = np.einsum('i,i...', self.BT[6,:6], Fs_x[:6])
        _NewData_x = self.Proj.P(Data_x, Step, direction_x)
        direction_x = np.einsum('i,i...', self.BT[5,:6], Fs_x[:6])
        NewData_x = self.Proj.P(Data_x, Step, direction_x)

        # Compute order-p, order-p+1 psi
        direction_psi = np.einsum('i,i...', self.BT[6,:6], Fs_psi[:6])
        _NewData_psi = Data_psi+Step*direction_psi
        direction_psi = np.einsum('i,i...', self.BT[5,:6], Fs_psi[:6])
        NewData_psi = Data_psi+Step*direction_psi

        # Compute Deltas
        Delta_x = max(abs(NewData_x-_NewData_x))
        Delta_psi = max(abs(NewData_psi-_NewData_psi))

        # Orthogonalize Psi, Record Lyapunov Exponents, Normalize Psi
        NewData_psi = NewData_psi.reshape((dim,-1))
        NewData_psi = GramSchmidt(NewData_psi,normalize=False)
        LEDT = np.log(np.linalg.norm(NewData_psi,axis=0))
        Tnew = T + abs(Step)
        NewLyapunov = (Lyapunov*T+LEDT)/Tnew
        NewData_psi = NewData_psi/np.linalg.norm(NewData_psi,axis=0)

        # Adjust Stepsize
        Delta = max(Delta_x,Delta_psi)
        if Delta == 0:
            growth = self.GrowthLimit
        else:
            growth = min((self.Delta0/Delta)**0.2, self.GrowthLimit)
        Step = np.clip(growth*Step,self.MinStep,self.MaxStep)

        # Store Data
        TempData['Data'] = NewData_x
        TempData[self.F] = self.F(NewData_x)
        TempData['Psi'] = NewData_psi.flatten()
        TempData['dPsi'] = np.dot(self.Jac(NewData_x),NewData_psi).flatten()
        TempData['Lyapunov'] = NewLyapunov
        TempData['T'] = Tnew
        TempData['Step'] = Step
        TempData['F Evaluations'] = 6 + self.TempStorage['F Evaluations'][-1]
        TempData['Projections'] = 6 + self.TempStorage['Projections'][-1]
        self.BookKeeping(TempData)

        return self.TempStorage
