__author__ = 'clemens'

import numpy as np

from VISolver.Domains.Domain import Domain


class CostFunctions():
    """
    A class collecting all the cost functions
    """
    @staticmethod
    def linear_cost_function(traffic_flow, base_cost, capacity=1e6, add_cost=.001):
        if traffic_flow < capacity:
            return base_cost + (traffic_flow * add_cost)
        else:
            return 1e6

    @staticmethod
    def realistic_cost_function(traffic_flow, base_cost, capacity, mult_offset=.01, exp_offset=1.5):
        """
        This computes the cost of travelling on the road following an empirical model taken from the Bachelor's thesis
        'Variational Inequalities and Complementarity Problems' by Friedemann Sebastian Winkler.
        """
        return base_cost * (1 + mult_offset * (traffic_flow/capacity)**exp_offset)


class NetworkNode():
    """
    One element of the network; resembling a cross section of the traffic map
    """
    def __init__(self, ident):
        self.id = ident
        self.layer_from_start = -1
        # the successors should have the format
        #   id: [(node) successor, base_cost, capacity, usage, cost, (opt) offs1, (opt) offs2]
        self.successors = {}


class Network():
    """
    the class storing the information on the traffic network problem
    """
    def __init__(self, no_nodes, max_path_length, max_edge_length, path_range, sparsity=.5, max_capacity=1e3,
                 b_dead_end=True, b_one_way=True, cf=CostFunctions.linear_cost_function):
        """
        This function initializes the network for the traffic problem
        :param max_path_length:
        :param no_nodes:
        :param max_edge_length:
        :param path_range:
        :param sparsity:
        :param max_capacity:
        :param b_dead_end:
        :param b_one_way:
        :return:
        """
        self.no_nodes = no_nodes
        self.max_path_length = max_path_length
        self.max_edge_length = max_edge_length
        self.path_range = path_range
        self.sparsity = sparsity
        self.max_capacity = max_capacity
        self.has_dead_ends = b_dead_end
        self.is_reversible = not b_one_way
        # initialize the set of nodes
        self.nodes = [NetworkNode(i) for i in range(self.no_nodes)]
        # initializing the layer of the first node
        self.nodes[0].layer_from_start = 0
        self.nodex_by_id = {}
        for i in range(self.no_nodes):
            self.nodex_by_id[i] = self.nodes[i]
        self.create_connections()
        # each edge stores length, capacity, usage, cost
        self.adjacency_matrix = 1e6*np.ones((self.no_nodes, self.no_nodes, 4))
        self.cost_function = cf

    def compute_path_cost(self, index_succ, flow):
        """
        This method computes the cost of taking one of the paths
        """
        return self.cost_function(flow, self.successors[index_succ][1], self.successors[index_succ][2])

    def init_single_connection(self, origin, target, length=None, capacity=None, usage=0):
        """
        This method creates a single connection
        :param origin:
        :param target:
        :param length:
        :param capacity:
        :return:
        """
        if length is None:
            length = self.max_edge_length*np.random.random()
        if capacity is None:
            capacity = self.max_capacity*np.random.random()
        cost = self.cost_function(length, capacity, usage)
        self.nodes[origin].successors[target] = [self.nodes[target],
                                                 length,
                                                 capacity,
                                                 usage,
                                                 cost]
        self.adjacency_matrix[origin, target, :] = [length, capacity, usage, cost]
        if self.nodes[target].layer_from_start == -1:
            self.nodes[target].layer_from_start = self.nodes[origin].layer_from_start + 1

    def update_single_connection(self, origin, target, diff=0, new_total=None, update_all=False):
        """
        adf
        :param diff:
        :param new_total:
        :return:
        """
        if new_total is not None:
            self.nodes[origin].successors[target][3] = new_total
            self.adjacency_matrix[origin, target, 2] = new_total
        else:
            self.nodes[origin].successors[target][3] += diff
            self.adjacency_matrix[origin, target, 2] += diff

        if update_all:
            length, capacity, usage = self.adjacency_matrix[origin, target, :-1]
            new_cost = self.cost_function(length, capacity, usage)
            self.nodes[origin].successors[target][4] = new_cost
            self.adjacency_matrix[origin, target, 3] = new_cost

    def update_all_costs(self):
        """
        This function updates the costs for all the edges.
        :return:
        """
        for i in self.no_nodes:
            for j in self.no_nodes:
                if j in self.nodes[i].successors:
                    self.nodes[i].successors[j][4] = self.cost_function(self.nodes[i].successors[j][0],
                                                                        self.nodes[i].successors[j][1],
                                                                        self.nodes[i].successors[j][2],
                                                                        self.nodes[i].successors[j][3])
                    self.adjacency_matrix[i, j, 3] = self.nodes[i].successors[j][4]

    def create_connections(self):
        """
        The method constructing the paths (i.e. "streets")
        """
        # initializing with one direct path from start to finish, to guarantee one path (however, this path has minimal
        # capacity). The path may be overriden later on.
        self.init_single_connection(0, self.no_nodes-1, self.no_nodes*self.max_edge_length, 1e-2)
        if not self.has_dead_ends:
            for in_node in range(1, self.no_nodes-1, 1):
                self.init_single_connection(in_node, self.no_nodes-1, self.max_edge_length, 1e-2)
        for in_node in range(self.no_nodes):
            if self.is_reversible:
                target_nodes = range(self.no_nodes)
            else:
                target_nodes = range(in_node, self.no_nodes, 1)
            for in_node_2 in target_nodes:
                if in_node != in_node_2 and \
                                np.random.random() > self.sparsity and \
                                self.nodes[in_node].layer_from_start < self.max_path_length:
                    self.init_single_connection(in_node, in_node_2)


class TrafficNetwork(Domain):
    """
    The matching pennies domain (http://en.wikipedia.org/wiki/Matching_pennies)
    """
    def __init__(self):
        self.players = 2
        self.reward_range = [-1, 1]
        self.dim = 2
        self.r_reward = np.array([[1., -1.], [-1., -1]])
        self.c_reward = np.array([[1., -1.], [-1., -1.]])
        self.u = self.u()
        self.uprime = self.uprime()
        self.A = np.array([[0., self.u], [self.uprime, 0.]])
        self.b = np.array(
            [-(self.r_reward[1, 1] - self.r_reward[0, 1]), -(self.c_reward[1, 1] - self.c_reward[1, 0])])
        self.A_curl = np.array(
            [[2. * self.uprime ** 2., 0], [0, 2. * self.u ** 2.]])
        self.b_curl = np.array([-2. * self.uprime * (self.c_reward[1, 1] - self.c_reward[1, 0]),
                                -2. * self.u * (self.r_reward[1, 1] - self.r_reward[0, 1])])
        self.NE = np.array([[1., .0],
                            [1., .0]])  # 1 mixed NE

    def u(self):
        return (self.r_reward[0, 0] + self.r_reward[1, 1]) - (self.r_reward[1, 0] + self.r_reward[0, 1])

    def uprime(self):
        return (self.c_reward[0, 0] + self.c_reward[1, 1]) - (self.c_reward[1, 0] + self.c_reward[0, 1])

    def f(self, data):
        return self.A.dot(data) + self.b

    def f_curl(self, data):
        return 0.5 * self.A_curl.dot(data) + self.b_curl

    def ne_l2error(self, data):
        return np.linalg.norm(data - self.NE)

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