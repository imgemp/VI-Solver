from VISolver.Projection import *
from Solver import Solver
from VISolver.Utilities import *


class DriftABE5(Solver):

    def __init__(self, domain, proj=IdentityProjection(), delta0=1e-2, growth_limit=2, min_step=-1e10, max_step=1e10):

        self.R = [domain.r, domain.c]
        self.Proj = proj
        self.StorageSize = 50
        self.temp_storage = {}
        self.Delta0 = delta0
        self.GrowthLimit = growth_limit
        self.MinStep = min_step
        self.MaxStep = max_step
        self.Mod = 200  # (100)
        self.Agg = 1.  # (10)
        self.Stall = 10
        self.agent_i = 0
        self.goodbad = [0, 0]

    def init_temp_storage(self, Start, Domain, Options):

        self.temp_storage['Policy']                 = self.StorageSize * [Start]
        self.temp_storage['Value Function']         = self.StorageSize * [np.zeros(Domain.r.shape)]
        self.temp_storage['Policy Gradient (dPi)']  = self.StorageSize * [np.zeros(Domain.b.shape)]
        self.temp_storage['Policy Learning Rate']   = self.StorageSize * [Options.Init.Step]
        self.temp_storage['Value Learning Rate']    = self.StorageSize * [0.1]
        self.temp_storage['dPi.dPi']                = self.StorageSize * \
                                                      [np.sum(self.temp_storage['Policy Gradient (dPi)'][-1] ** 2)]
        self.temp_storage['Projections']            = self.StorageSize * [0]

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
        Pi = record.temp_storage['Policy'][-1]
        V = record.temp_storage['Value Function'][-1]
        dPi = record.temp_storage['Policy Gradient (dPi)'][-1]
        Eta = record.temp_storage['Policy Learning Rate'][-1]
        Alpha = record.temp_storage['Value Learning Rate'][-1]

        # Initialize Storage
        TempData = {}

        # May want to use a Finite State Machine approach rather than if
        # statements

        if (record.this_perm_index >= self.Mod) and (
                record.this_perm_index % self.Mod < self.Stall):

            # Perform Euler update on Policies and Project onto Simplex
            Pi_1 = self.Proj.P(Pi[0], Eta, dPi[0])  # Player 1
            Pi_2 = self.Proj.P(Pi[1], Eta, dPi[1])  # Player 2
            Pi_New = np.array([Pi_1, Pi_2])

            # Record Projections
            TempData['Projections'] = 1 + self.temp_storage['Projections'][-1]

        elif (record.this_perm_index >= self.Mod) and (record.this_perm_index % self.Mod == self.Stall):

            # Approximate Gradient of G = ||dPi||^2 with respect to agent i
            G_stat_plus = np.array(
                self.temp_storage['dPi.dPi'][-self.Stall + 1:])
            G_stat_minus = np.array(
                self.temp_storage['dPi.dPi'][-self.Stall:-1])
            G_dyn_plus = np.array(
                self.temp_storage['dPi.dPi'][-2 * self.Stall + 1:-self.Stall])
            G_dyn_minus = np.array(
                self.temp_storage['dPi.dPi'][-2 * self.Stall:-self.Stall - 1])

            x_dyn_plus = np.array(
                self.temp_storage['Policy'][-2 * self.Stall + 1:-self.Stall][self.agent_i])
            x_dyn_minus = np.array(
                self.temp_storage['Policy'][-2 * self.Stall:-self.Stall - 1][self.agent_i])

            dG_stat = np.mean(G_stat_plus - G_stat_minus)
            dG_dyn = np.mean(G_dyn_plus - G_dyn_minus)
            dx_dyn = np.mean(x_dyn_plus[:, 0] - x_dyn_minus[:, 0])

            dG_dxi = (dG_dyn - dG_stat) / dx_dyn

            # print('Current Policy:');
            # print('Player 1'); print(Pi[0])
            # print('Player 2'); print(Pi[1])

            # print('dG_dxi'); print(dG_dxi)
            # print('dG_dyn'); print(dG_dyn)
            # print('dG_stat'); print(dG_stat)
            # print('dx_dyn'); print(dx_dyn)

            if np.sign(dG_dxi) == np.sign(Pi[self.agent_i][0] - .5):
                self.goodbad[0] += 1
            else:
                self.goodbad[1] += 1
            # Compute Adjusted Policy Gradient
            dPi_norm = np.linalg.norm(dPi[self.agent_i])
            dG_dxi_norm = np.linalg.norm(dG_dxi)
            # print(`self.Agg*0.5*dG_dxi*dPi_norm/dG_dxi_norm`+'\tcurl component')
            dPi[self.agent_i][0] += - self.Agg * \
                0.5 * dG_dxi * dPi_norm / dG_dxi_norm
            # print(`dPi[self.agent_i]`+'\tnew dPi of agent_i'); print(`self.agent_i`+'\tagent_i');
            # Perform Euler update on Policies and Project onto Simplex
            Pi_1 = self.Proj.P(Pi[0], Eta, dPi[0])  # Player 1
            Pi_2 = self.Proj.P(Pi[1], Eta, dPi[1])  # Player 2
            # print('Player 1'); print(dPi[0]); print(Pi_1);
            # print('Player 2'); print(dPi[1]); print(Pi_2);
            Pi_New = np.array([Pi_1, Pi_2])
            # print(`Pi[self.agent_i]`+'\toriginal policy of agent_i'); print(`Pi_New[self.agent_i]`+'\tnew policy of agent_i'); print('-----------')
            # Record Projections
            TempData['Projections'] = 1 + self.temp_storage['Projections'][-1]

        else:

            if (record.this_perm_index %
                2 == 0) or (record.this_perm_index %
                            self.Mod == self.Stall + 1):

                # Perform Euler update on Policies and Project onto Simplex
                Pi_1 = self.Proj.P(Pi[0], Eta, dPi[0])  # Player 1
                Pi_2 = self.Proj.P(Pi[1], Eta, dPi[1])  # Player 2
                Pi_New = np.array([Pi_1, Pi_2])

                # Record Projections
                TempData['Projections'] = 1 + \
                    self.temp_storage['Projections'][-1]

            else:

                # Perform Adams Bashforth update
                dPis = record.temp_storage['Policy Gradient (dPi)']
                Pi_1 = self.Proj.P(
                    Pi[0], Eta, -0.5 * dPis[-2][0] + 1.5 * dPis[-1][0])
                Pi_2 = self.Proj.P(
                    Pi[1], Eta, -0.5 * dPis[-2][1] + 1.5 * dPis[-1][1])
                Pi_New = np.array([Pi_1, Pi_2])

                # Perform Euler update
                _Pi_1 = self.Proj.P(Pi[0], Eta, dPis[-1][0])  # Player 1
                _Pi_2 = self.Proj.P(Pi[1], Eta, dPis[-1][1])  # Player 2
                _Pi_New = np.array([_Pi_1, _Pi_2])

                # Adjust Stepsize
                Delta = np.max(abs(Pi_New - _Pi_New))
                if Delta == 0.:
                    Eta = max(min(Eta * 2., self.MaxStep), self.MinStep)
                else:
                    Eta = max(min(Eta * min((self.Delta0 / Delta) ** 0.5,
                                            self.GrowthLimit),
                                  self.MaxStep),
                              self.MinStep)

                # Record Projections
                TempData['Projections'] = 2 + \
                    self.temp_storage['Projections'][-1]
        # print('Policy'); print(Pi_New[0]); print(Pi_New[1])
        # Play Game
        # Select Actions According to Policies
        a_1 = self.Action(Pi_New[0])  # Player 1
        a_2 = self.Action(Pi_New[1])  # Player 2
        # print('actions')
        # print(a_1)
        # print(a_2)

        # Play those actions and observe the resulting rewards
        r_1 = self.R[0][a_1, a_2]  # Player 1
        r_2 = self.R[1][a_1, a_2]  # Player 2
        # print('rewards')
        # print(r_1)
        # print(r_2)

        # update Value Function
        V_New = np.array(V)
        V_New[0][a_1] = Alpha * r_1 + (1 - Alpha) * V[0][a_1]  # Player 1
        V_New[1][a_2] = Alpha * r_2 + (1 - Alpha) * V[1][a_2]  # Player 2
        # V_New = np.array([V_1,V_2])
        # print('V')
        # print(V_New[0])
        # print(V_New[1])

        # Compute Total Average Reward
        TV_1 = np.sum(V_New[0] / V_New[0].size)
        TV_2 = np.sum(V_New[1] / V_New[1].size)

        # Compute Policy Gradients
        dPi_1 = V_New[0] - TV_1  # Player 1
        dPi_2 = V_New[1] - TV_2  # Player 2
        dPi_New = np.array([dPi_1, dPi_2])

        if (record.this_perm_index >= self.Mod -
            1) and (record.this_perm_index %
                    self.Mod < self.Stall -
                    1):
            # Choose Agent for Curl Component
            self.agent_i = 0  # np.random.randint(Pi.shape[0])
            # Freeze Agent i's Policy
            dPi_New[self.agent_i] = 0. * dPi[self.agent_i]

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
        TempData['dPi.dPi'] = np.sum(dPi_New ** 2)
        self.book_keeping(TempData)

        return self.temp_storage
