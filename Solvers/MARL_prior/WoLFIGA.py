__author__ = 'clemens'

from VISolver.Projection import *
from VISolver.Solvers.Solver import Solver
from VISolver.Utilities import *
from VISolver.Options import Reporting


class WoLFIGA(Solver):
    def __init__(self, domain, P=IdentityProjection(), delta0=1e-2, growth_limit=2, min_step=1e-5, max_step=1e-3):

        self.reward = [domain.r_reward, domain.c_reward]
        self.domain = domain
        self.nash_eq_value = domain.compute_nash_eq_value()
        self.Proj = P
        self.storage_size = 3
        self.temp_storage = {}
        self.MinStep = min_step
        self.MaxStep = max_step

    def init_temp_storage(self, start, domain, options):

        self.temp_storage['Policy'] = self.storage_size * [start]
        self.temp_storage['Value Function'] = self.storage_size * [np.zeros(domain.r_reward.shape)]
        self.temp_storage['Policy Gradient (dPi)'] = self.storage_size * [np.zeros(domain.b.shape)]
        self.temp_storage['Policy Learning Rate'] = self.storage_size * [[options.init.step], [options.init.step]]
        self.temp_storage['Projections'] = self.storage_size * [0]

        return self.temp_storage

    def reporting_options(self):
        return Reporting(requests=[self.domain.ne_l2error,
                                   'Policy',
                                   'Policy Gradient (dPi)',
                                   'Policy Learning Rate',
                                   'Reward',
                                   'Value Function',
                                   'True Value Function',
                                   'Value Variance',
                                   'Performance',
                                   'Am I winning?',
                                   # 'Policy Estimates',
                                   ])

    def update(self, record):
        # Retrieve Necessary Data
        policy = record.temp_storage['Policy'][-1]
        learn_min = self.MinStep
        learn_max = self.MaxStep

        # Initialize Storage
        temp_data = {}

        # compute the value of the current strategy
        value = self.domain.compute_value_function(policy)

        # decide on the learning rate
        learning_rate = [0, 0]
        learning_rate[0] = learn_min if value[0] > self.nash_eq_value[0] else learn_max
        learning_rate[1] = learn_min if value[1] > self.nash_eq_value[1] else learn_max

        # estimate the gradient
        policy_gradient = [ policy[1]*self.domain.u-     (self.reward[0][1][1]-self.reward[0][0][1]),
                            policy[0]*self.domain.uprime-(self.reward[1][1][1]-self.reward[1][1][0])]

        # perform update on the policies and project them into the feasible space
        updated_policy = [  self.Proj.p(policy[0], learning_rate[0], policy_gradient[0]),  # Player 1
                            self.Proj.p(policy[1], learning_rate[1], policy_gradient[1])]  # Player 2

        # Record Projections
        temp_data['Projections'] = 1 + self.temp_storage['Projections'][-1]

        # Store Data
        temp_data['Policy'] = updated_policy
        temp_data['Value Function'] = value
        temp_data['Policy Gradient (dPi)'] = policy_gradient
        temp_data['Policy Learning Rate'] = learning_rate
        self.book_keeping(temp_data)

        return self.temp_storage