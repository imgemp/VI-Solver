import numpy as np

from Projection import *
from Utilities import *
from Solver import Solver

class DriftABE2(Solver):

    def __init__(self,Domain,P=IdentityProjection(),Delta0=1e-2,GrowthLimit=2,MinStep=-1e10,MaxStep=1e10):
        
        self.R = [Domain.r,Domain.c]

        self.Proj = P

        self.StorageSize = 3

        self.TempStorage = {}

        self.Delta0 = Delta0

        self.GrowthLimit = GrowthLimit

        self.MinStep = MinStep

        self.MaxStep = MaxStep

        self.Mod = 1e6 #(100)

        self.Agg = 1 #(10)

        self.agent_i = 0

    def InitTempStorage(self,Start,Domain,Options):

        self.TempStorage['Policy'] = self.StorageSize*[Start]
        self.TempStorage['Value Function'] = self.StorageSize*[np.zeros(Domain.r.shape)]
        self.TempStorage['Policy Gradient (dPi)'] = self.StorageSize*[np.zeros(Domain.b.shape)]
        self.TempStorage['Policy Learning Rate'] = self.StorageSize*[Options.Init.Step]
        self.TempStorage['Value Learning Rate'] = self.StorageSize*[0.1]
        self.TempStorage['dPi.dPi'] = self.StorageSize*[np.sum(self.TempStorage['Policy Gradient (dPi)'][-1]**2)]
        self.TempStorage['Projections'] = self.StorageSize*[0]

        return self.TempStorage

    # BookKeeping(self,TempData) defined in super class 'Solver'

    def Action(self,Policy):

        ind = np.random.rand()
        if (ind <= Policy[0]):
            return 0
        else:
            return 1

    def Update(self,Record):

        # Retrieve Necessary Data
        Pi = Record.TempStorage['Policy'][-1]
        V = Record.TempStorage['Value Function'][-1]
        dPi = Record.TempStorage['Policy Gradient (dPi)'][-1]
        Eta = Record.TempStorage['Policy Learning Rate'][-1]
        Alpha = Record.TempStorage['Value Learning Rate'][-1]

        # Initialize Storage
        TempData = {}

        # May want to use a Finite State Machine approach rather than if statements

        # if (Record.thisPermIndex>=self.Mod) and (Record.thisPermIndex%self.Mod == 0):

        #     # Choose Agent for Curl Component
        #     self.agent_i = np.random.randint(Pi.shape[0])

        #     # Freeze Agent i's Policy
        #     dV[self.agent_i] = 0

        #     # Perform Euler Update on Policies and Project onto Simplex
        #     Pi_1 = self.Proj.P(Pi[0],Eta,dV[0]) # Player 1
        #     Pi_2 = self.Proj.P(Pi[1],Eta,dV[1]) # Player 2
        #     Pi_New = np.array([Pi_1,Pi_2])

        #     # Record Projections
        #     TempData['Projections'] = 1 + self.TempStorage['Projections'][-1]

        # elif (Record.thisPermIndex>=self.Mod) and (Record.thisPermIndex%self.Mod == 1):

        #     # Approximate Gradient of ||dV||^2 with respect to agent i
        #     G_k = self.TempStorage['dV.dV'][-1]
        #     G_km1 = self.TempStorage['dV.dV'][-2]
        #     G_km2 = self.TempStorage['dV.dV'][-3]
        #     x_km1 = self.TempStorage['Policy'][-2][self.agent_i]
        #     x_km2 = self.TempStorage['Policy'][-3][self.agent_i]
        #     dG_dxi = (-G_k+2*G_km1-G_km2)/np.linalg.norm(x_km1-x_km2) # Policies are multidimensional, not sure what to do

        #     # Compute Adjusted Temporal Difference
        #     dV[self.agent_i] = dV[self.agent_i] - self.Agg*0.5*dG_dxi

        #     # Perform Euler Update on Policies and Project onto Simplex
        #     Pi_1 = self.Proj.P(Pi[0],Eta,dV[0]) # Player 1
        #     Pi_2 = self.Proj.P(Pi[1],Eta,dV[1]) # Player 2
        #     Pi_New = np.array([Pi_1,Pi_2])

        #     # Record Projections
        #     TempData['Projections'] = 1 + self.TempStorage['Projections'][-1]

        # else:

            # if (Record.thisPermIndex%2 == 0):

        # Perform Euler Update on Policies and Project onto Simplex
        Pi_1 = self.Proj.P(Pi[0],Eta,dPi[0]) # Player 1
        Pi_2 = self.Proj.P(Pi[1],Eta,dPi[1]) # Player 2
        Pi_New = np.array([Pi_1,Pi_2])

        # Record Projections
        TempData['Projections'] = 1 + self.TempStorage['Projections'][-1]

            # else:

            #     # Perform Adams Bashforth Update
            #     dVs = Record.TempStorage['Temporal Difference (dV)']
            #     Pi_1 = self.Proj.P(Pi[0],Eta,-0.5*dVs[0][-2]+1.5*dVs[0][-1])
            #     Pi_2 = self.Proj.P(Pi[1],Eta,-0.5*dVs[1][-2]+1.5*dVs[1][-1])
            #     Pi_New = np.array([Pi_1,Pi_2])

            #     # Perform Euler Update
            #     _Pi_1 = self.Proj.P(Pi[0],Eta,dVs[0][-1]) # Player 1
            #     _Pi_2 = self.Proj.P(Pi[1],Eta,dVs[1][-1]) # Player 2
            #     _Pi_New = np.array([_Pi_1,_Pi_2])

            #     # Adjust Stepsize
            #     Delta = np.max(abs(Pi_New-_Pi_New))
            #     if Delta == 0.: Eta = 2.*Eta
            #     else: Eta = max(min(Eta*min((self.Delta0/Delta)**0.5,self.GrowthLimit),self.MaxStep),self.MinStep)

            #     # Record Projections
            #     TempData['Projections'] = 2 + self.TempStorage['Projections'][-1]
        # print('Policy'); print(Pi_New[0]); print(Pi_New[1])
        # Play Game
        # Select Actions According to Policies
        a_1 = self.Action(Pi_New[0]) # Player 1
        a_2 = self.Action(Pi_New[1]) # Player 2
        # print('actions')
        # print(a_1)
        # print(a_2)

        # Play those actions and observe the resulting rewards
        r_1 = self.R[0][a_1,a_2] # Player 1
        r_2 = self.R[1][a_1,a_2] # Player 2
        # print('rewards')
        # print(r_1)
        # print(r_2)

        # Update Value Function
        V_New = np.array(V)
        V_New[0][a_1] = Alpha*r_1 + (1-Alpha)*V[0][a_1] # Player 1
        V_New[1][a_2] = Alpha*r_2 + (1-Alpha)*V[1][a_2] # Player 2
        # V_New = np.array([V_1,V_2])
        # print('V')
        # print(V_New[0])
        # print(V_New[1])

        # Compute Total Average Reward
        TV_1 = np.sum(V_New[0]/V_New[0].size)
        TV_2 = np.sum(V_New[1]/V_New[1].size)

        # Compute Policy Gradients
        dPi_1 = V_New[0] - TV_1 # Player 1
        dPi_2 = V_New[1] - TV_2 # Player 2
        dPi_New = np.array([dPi_1,dPi_2])
        # print('dPi')
        # print(dPi_1)
        # print(dPi_2)
        # print('--------------------------------------')
        # Store Data
        TempData['Policy'] = Pi_New
        TempData['Value Function'] = V_New
        TempData['Policy Gradient (dPi)'] = dPi_New
        TempData['Policy Learning Rate'] = Eta
        TempData['Value Learning Rate'] = Alpha
        TempData['dPi.dPi'] = np.sum(dPi_New**2)
        self.BookKeeping(TempData)
        
        return self.TempStorage





