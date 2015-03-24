import numpy as np

from VISolver.Domains.Domain import Domain


class PrisonersDilemma(Domain):

    def __init__(self):
        self.players = 2
        self.reward_range = [0, 3]
        self.Dim = 2
        self.r_reward = np.array([[2., -1.], [3., 0.]])
        self.c_reward = np.array([[2., 3.], [-1., 0.]])
        self.u = self.u()
        self.uprime = self.uprime()
        self.A = np.array([[0., self.u], [self.uprime, 0.]])
        self.b = np.array(
            [-(self.r_reward[1, 1] - self.r_reward[0, 1]), -(self.c_reward[1, 1] - self.c_reward[1, 0])])
        self.A_curl = np.array(
            [[2. * self.uprime ** 2., 0], [0, 2. * self.u ** 2.]])
        self.b_curl = np.array([-2. * self.uprime * (self.c_reward[1, 1] - self.c_reward[1, 0]),
                                -2. * self.u * (self.r_reward[1, 1] - self.r_reward[0, 1])])
        self.NE = np.array([1., 1.])  # 1 mixed NE

    def u(self):
        return (self.r_reward[0, 0] + self.r_reward[1, 1]) - (self.r_reward[1, 0] + self.r_reward[0, 1])

    def uprime(self):
        return (self.c_reward[0, 0] + self.c_reward[1, 1]) - (self.c_reward[1, 0] + self.c_reward[0, 1])

    def F(self, Data):
        return self.A.dot(Data) + self.b

    def F_curl(self, Data):
        return 0.5 * self.A_curl.dot(Data) + self.b_curl

    def ne_l2error(self, Data):
        return np.linalg.norm(Data - self.NE)

    def compute_value_function(self, policy, policy_approx=None):
        if policy_approx is None:
            policy_approx = policy
        value = [0., 0.]
        # computing the first player's value function -> relying on estimates for the second player's strategy
        value[0]= self.r_reward[0][0]*policy[0][0]*policy_approx[1][0]\
                + self.r_reward[1][1]*policy[0][1]*policy_approx[1][1]\
                + self.r_reward[0][1]*policy[0][0]*policy_approx[1][1]\
                + self.r_reward[1][0]*policy[0][1]*policy_approx[1][0]

        # computing the second player's value function -> relying on estimates for the first player's strategy
        value[1]= self.c_reward[0][0]*policy_approx[0][0]*policy[1][0]\
                + self.c_reward[1][1]*policy_approx[0][1]*policy[1][1]\
                + self.c_reward[0][1]*policy_approx[0][0]*policy[1][1]\
                + self.c_reward[1][0]*policy_approx[0][1]*policy[1][0]
        return value

    def compute_nash_eq_value(self):
        policy = [[self.NE[0], 1-self.NE[0]],
                  [self.NE[1], 1-self.NE[1]]]
        return self.compute_value_function(policy)

    @staticmethod
    def action(policy):
        ind = np.random.rand()
        try:
            if ind <= policy[0]:
                return 0
            else:
                return 1
        except ValueError:
            if ind <= policy[0][0]:
                return 0
            else:
                return 1