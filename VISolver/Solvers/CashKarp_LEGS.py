import numpy as np

from VISolver.Projection import IdentityProjection
from VISolver.Solver import Solver


class CashKarp_LEGS(Solver):

    def __init__(self, Domain, P=IdentityProjection(), Delta0=1e-4,
                 Delta0_L=1e-4, GrowthLimit=2, MinStep=-1e10, MaxStep=1e10):

        self.F = Domain.F

        self.Jac = Domain.Jac

        self.Proj = P

        self.StorageSize = 1

        self.TempStorage = {}

        self.Delta0 = Delta0

        self.Delta0_L = Delta0_L

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
        self.TempStorage['Step'] = self.StorageSize*[Options.Init.Step]

        Psi_0 = np.eye(Start.size)
        dPsi_0 = np.dot(self.Jac(Start),Psi_0)
        self.TempStorage['Psi'] = self.StorageSize*[Psi_0.flatten()]
        self.TempStorage['dPsi'] = self.StorageSize*[dPsi_0.flatten()]
        self.TempStorage['Lyapunov'] = self.StorageSize*[0*Start]
        self.TempStorage['L-Step'] = self.StorageSize*[Options.Init.Step]

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
        dim = Data_x.size

        Fs_x = np.zeros((6,Data_x.shape[0]))
        Fs_x[0,:] = Record.TempStorage[self.F][-1]

        Fs_psi = np.zeros((2,Data_psi.shape[0]))
        Fs_psi[0,:] = Record.TempStorage['dPsi'][-1]
        # F_psi = Record.TempStorage['dPsi'][-1]

        Step = Record.TempStorage['Step'][-1]
        Step_L = Record.TempStorage['L-Step'][-1]

        # Initialize Storage
        TempData = {}

        # Update Psi
        # Step_L = .00001
        NewData_psi = Data_psi+Step_L*Fs_psi[0,:]
        NewData_psi_rsh = NewData_psi.reshape((dim,-1))
        _NewData_x = self.Proj.P(Data_x, Step_L, Fs_x[0,:])
        Fs_psi[1,:] = np.dot(self.Jac(_NewData_x),NewData_psi_rsh).flatten()
        _NewData_psi = Data_psi+Step_L*0.5*np.sum(Fs_psi,axis=0)

        # Record Lyapunov exponent and Orthonormalize Psi
        NewLyapunov = np.log(np.linalg.norm(NewData_psi_rsh,axis=0))*Step/Step_L
        Psi, R = np.linalg.qr(NewData_psi_rsh)

        # Adjust Stepsize
        Delta = max(abs(NewData_psi-_NewData_psi))
        if Delta == 0:
            Step_L = self.GrowthLimit * Step_L
        else:
            growth = min((self.Delta0_L/Delta)**0.5, self.GrowthLimit)
            Step_L = np.clip(Step_L*growth, self.MinStep, self.MaxStep)

        # Calculate k values (gradients)
        for i in xrange(5):
            direction_x = np.einsum('i,i...', self.BT[i,:i+1], Fs_x[:i+1])
            # direction_psi = np.einsum('i,i...', self.BT[i,:i+1], Fs_psi[:i+1])

            _NewData_x = self.Proj.P(Data_x, Step, direction_x)
            # _NewData_psi = (Data_psi+Step*direction_psi).reshape((dim,-1))

            Fs_x[i+1,:] = self.F(_NewData_x)
            #Fs_psi[i+1,:] = np.dot(self.Jac(_NewData_x),_NewData_psi).flatten()

        # Compute order-p, order-p+1 data points
        direction_x = np.einsum('i,i...', self.BT[6,:6], Fs_x[:6])
        _NewData_x = self.Proj.P(Data_x, Step, direction_x)
        direction_x = np.einsum('i,i...', self.BT[5,:6], Fs_x[:6])
        NewData_x = self.Proj.P(Data_x, Step, direction_x)

        # Compute order-p+1 for psi
        # direction_psi = np.einsum('i,i...', self.BT[5,:6], Fs_psi[:6])
        # NewData_psi = Data_psi+Step*direction_psi
        # NewData_psi = Data_psi+.0001*F_psi
        # Psi = NewData_psi.reshape((dim,-1))

        # # Record Lyapunov exponent and Orthonormalize Psi
        # NewLyapunov = np.log(np.linalg.norm(Psi,axis=0))/.0001
        # Psi, R = np.linalg.qr(Psi)

        # Adjust Stepsize
        Delta = max(abs(NewData_x-_NewData_x))
        if Delta == 0.:
            Step = self.GrowthLimit * Step
        else:
            growth = min(self.GrowthLimit, (self.Delta0/Delta)**0.2)
            Step = np.clip(Step*growth,self.MinStep,self.MaxStep)

        # Store Data
        TempData['Data'] = NewData_x
        TempData[self.F] = self.F(NewData_x)
        TempData['Psi'] = Psi.flatten()
        TempData['dPsi'] = np.dot(self.Jac(NewData_x),Psi).flatten()
        TempData['Lyapunov'] = Lyapunov + NewLyapunov
        TempData['Step'] = Step
        TempData['L-Step'] = Step_L
        TempData['F Evaluations'] = 6 + self.TempStorage['F Evaluations'][-1]
        TempData['Projections'] = 6 + self.TempStorage['Projections'][-1]
        self.BookKeeping(TempData)

        return self.TempStorage
