import numpy as np

from Projection import *
from Utilities import *
from Solver import Solver


class DriftABE_VIteration(Solver):

    def __init__(
            self,
            Domain,
            P=IdentityProjection(),
            Delta0=1e-2,
            GrowthLimit=2,
            MinStep=-1e10,
            MaxStep=1e10):

        self.R = [Domain.r, Domain.c]

        self.Proj = P

        self.StorageSize = 3

        self.temp_storage = {}

        self.Delta0 = Delta0

        self.GrowthLimit = GrowthLimit

        self.MinStep = MinStep

        self.MaxStep = MaxStep

        self.Mod = 1e6  # (100)

        self.Agg = 1  # (10)

        self.agent_i = 0

    def init_temp_storage(self, Start, Domain, Options):

        self.temp_storage['Policy'] = self.StorageSize * [Start]
        self.temp_storage[
            'Value Function'] = self.StorageSize * [np.zeros(Domain.r.shape)]
        self.temp_storage[
            'Temporal Difference (dV)'] = self.StorageSize * [np.zeros(Domain.b.shape)]
        self.temp_storage[
            'Policy Learning Rate'] = self.StorageSize * [Options.Init.Step]
        self.temp_storage['Value Learning Rate'] = self.StorageSize * [0.1]
        self.temp_storage['dV.dV'] = self.StorageSize * \
            [np.sum(self.temp_storage['Temporal Difference (dV)'][-1] ** 2)]
        self.temp_storage['Projections'] = self.StorageSize * [0]

        return self.temp_storage

    # book_keeping(self,TempData) defined in super class 'Solver'

    def Action(self, Policy):

        ind = np.random.rand()
        if ind <= Policy[0]:
            return 0
        else:
            return 1

    def update(self, record):

        # Retrieve Necessary Data
        Pi = record.TempStorage['Policy'][-1]
        V = record.TempStorage['Value Function'][-1]
        dV = record.TempStorage['Temporal Difference (dV)'][-1]
        Eta = record.TempStorage['Policy Learning Rate'][-1]
        Alpha = record.TempStorage['Value Learning Rate'][-1]

        # Initialize Storage
        TempData = {}

        # May want to use a Finite State Machine approach rather than if
        # statements

        if (record.thisPermIndex >= self.Mod) and (
                record.thisPermIndex % self.Mod == 0):

            # Choose Agent for Curl Component
            self.agent_i = np.random.randint(Pi.shape[0])

            # Freeze Agent i's Policy
            dV[self.agent_i] = 0

            # Perform Euler update on Policies and Project onto Simplex
            Pi_1 = self.Proj.P(Pi[0], Eta, dV[0])  # Player 1
            Pi_2 = self.Proj.P(Pi[1], Eta, dV[1])  # Player 2
            Pi_New = np.array([Pi_1, Pi_2])

            # Record Projections
            TempData['Projections'] = 1 + self.temp_storage['Projections'][-1]

        elif (record.thisPermIndex >= self.Mod) and (record.thisPermIndex % self.Mod == 1):

            # Approximate Gradient of ||dV||^2 with respect to agent i
            G_k = self.temp_storage['dV.dV'][-1]
            G_km1 = self.temp_storage['dV.dV'][-2]
            G_km2 = self.temp_storage['dV.dV'][-3]
            x_km1 = self.temp_storage['Policy'][-2][self.agent_i]
            x_km2 = self.temp_storage['Policy'][-3][self.agent_i]
            # Policies are multidimensional, not sure what to do
            dG_dxi = (-G_k + 2 * G_km1 - G_km2) / np.linalg.norm(x_km1 - x_km2)

            # Compute Adjusted Temporal Difference
            dV[self.agent_i] += - self.Agg * 0.5 * dG_dxi

            # Perform Euler update on Policies and Project onto Simplex
            Pi_1 = self.Proj.P(Pi[0], Eta, dV[0])  # Player 1
            Pi_2 = self.Proj.P(Pi[1], Eta, dV[1])  # Player 2
            Pi_New = np.array([Pi_1, Pi_2])

            # Record Projections
            TempData['Projections'] = 1 + self.temp_storage['Projections'][-1]

        else:

            # if (Record.thisPermIndex%2 == 0):

            # Perform Euler update on Policies and Project onto Simplex
            Pi_1 = self.Proj.P(Pi[0], Eta, dV[0])  # Player 1
            Pi_2 = self.Proj.P(Pi[1], Eta, dV[1])  # Player 2
            Pi_New = np.array([Pi_1, Pi_2])

            # Record Projections
            TempData['Projections'] = 1 + self.temp_storage['Projections'][-1]

            # else:

            # # Perform Adams Bashforth update
            #     dVs = Record.TempStorage['Temporal Difference (dV)']
            #     Pi_1 = self.Proj.P(Pi[0],Eta,-0.5*dVs[0][-2]+1.5*dVs[0][-1])
            #     Pi_2 = self.Proj.P(Pi[1],Eta,-0.5*dVs[1][-2]+1.5*dVs[1][-1])
            #     Pi_New = np.array([Pi_1,Pi_2])

            #     # Perform Euler update
            #     _Pi_1 = self.Proj.P(Pi[0],Eta,dVs[0][-1]) # Player 1
            #     _Pi_2 = self.Proj.P(Pi[1],Eta,dVs[1][-1]) # Player 2
            #     _Pi_New = np.array([_Pi_1,_Pi_2])

            #     # Adjust Stepsize
            #     Delta = np.max(abs(Pi_New-_Pi_New))
            #     if Delta == 0.: Eta = 2.*Eta
            # else: Eta =
            # max(min(Eta*min((self.Delta0/Delta)**0.5,self.GrowthLimit),self.max_step),self.min_step)

            #     # Record Projections
            #     TempData['Projections'] = 2 + self.TempStorage['Projections'][-1]
        print('Policy')
        print(Pi_New[0])
        print(Pi_New[1])
        # Play Game
        # Select Actions According to Policies
        a_1 = self.Action(Pi_New[0])  # Player 1
        a_2 = self.Action(Pi_New[1])  # Player 2
        print('actions')
        print(a_1)
        print(a_2)

        # Play those actions and observe the resulting rewards
        r_1 = self.R[0][a_1, a_2]  # Player 1
        r_2 = self.R[1][a_1, a_2]  # Player 2
        print('rewards')
        print(r_1)
        print(r_2)

        # update Value Function
        V_New = np.array(V)
        V_New[0][a_1] = Alpha * r_1 + (1 - Alpha) * V[0][a_1]  # Player 1
        V_New[1][a_2] = Alpha * r_2 + (1 - Alpha) * V[1][a_2]  # Player 2
        # V_New = np.array([V_1,V_2])
        print('V')
        print(V_New[0])
        print(V_New[1])

        # Compute Temporal Differences
        dV_1 = V_New[0] - V[0]  # Player 1
        dV_2 = V_New[1] - V[1]  # Player 2
        dV_New = np.array([dV_1, dV_2])
        print('dV')
        print(dV_1)
        print(dV_2)
        print('--------------------------------------')
        # Store Data
        TempData['Policy'] = Pi_New
        TempData['Value Function'] = V_New
        TempData['Temporal Difference (dV)'] = dV_New
        TempData['Policy Learning Rate'] = Eta
        TempData['Value Learning Rate'] = Alpha
        TempData['dV.dV'] = np.sum(dV_New ** 2)
        self.book_keeping(TempData)

        return self.temp_storage
