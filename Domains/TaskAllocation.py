__author__ = 'clemens'

import numpy as np
import random
from Domain import Domain


class TaskAllocation(Domain):
    """
    This domain is the task allocation problem as discussed by Abdallah and Lesser
    """
    def __init__(self, no_agents, no_providers, no_tasks_per_agent, cost_range_per_task):
        """
        The init function
        :param no_agents: the number of agents, i.e. consumers in the system
        :param no_providers: the number of providers in the system
        :param no_tasks_per_agent: the maximum number of tasks one agent want to solve (the value will be in [1, no]
        :param cost_range_per_task: the range of cost for each task (between 1 and x)
        :type no_agents: int
        :type no_providers: int
        :type no_tasks_per_agent: int
        :type cost_range_per_task: list
        :return:
        """
        self.props = [no_agents, no_providers, no_tasks_per_agent, cost_range_per_task]
        self.providers = [Provider() for _ in range(no_providers)]
        self.agents = [Agent(no_tasks_per_agent, cost_range_per_task, no_providers) for _ in range(no_agents)]

    def average_cost(self):
        """
        This function computes the current average cost of the model
        :return:
        """
        return sum([p.cost() for p in self.providers])/self.props[1]


class Agent:
    def __init__(self, no_tasks, range_of_tasks, providers):
        """
        The init function
        :param no_tasks: the maximum number of tasks; the exact value will be computed at random
        :param range_of_tasks: the cost range for each task
        :param providers: the providers in the system
        :type no_tasks: int
        :type range_of_tasks: list
        :type providers: list
        :return:
        """
        self.open_tasks = [random.randint(range_of_tasks[0], range_of_tasks[1]) for _ in range(random.randint(1, no_tasks))]
        self.providers = providers
        self.action_history = []
        self.cost = 0
        # the policy is the probability with which the agent assigns a taks to a provider. the 0th
        #       provider is the wait-step
        self.policy = [0.] + [1./len(providers) for _ in range(len(providers))]
        # self.policy = [0., 0., 1.]
        self.policy_history = []

    def play(self):
        """
        This function chooses an action according to the player's current policy
        :return: this round's reward
        :rtype: int
        """
        current_cost = self.cost
        action = self.sample_policy()
        if action == 0:
            self.wait()
        else:
            self.assign_task(action)
        return self.cost - current_cost

    def assign_task(self, provider_index):
        """
        This is an action, assigning the last task in the queue to the provider
        :param provider_index: the provider chosen
        :type provider_index: int
        :return: the provider's total workload after assignment
        :rtype: int
        """
        self.action_history.append(2)
        cost = self.providers[provider_index].assign(self.open_tasks.pop())
        self.cost -= cost
        return cost

    # def request_workload(self, provider):
    #     """
    #     This is an action, requesting the current workload of the provider
    #     :param provider: the provider
    #     :type provider: Provider
    #     :return: the provider's total workload
    #     """
    #     self.action_history.append(1)
    #     self.cost -= 1
    #     return provider.cost()

    def wait(self):
        """
        This is an action; if the stack is empty, the cost is 0. Otherwise, 1
        :return: None
        :rtype: None
        """
        self.action_history.append(0)
        if self.open_tasks:
            self.cost -= 1
        return None

    def sample_policy(self):
        """
        This function chooses an action given our distribution (i.e. policy)
        :return: the index chosen
        :rtype: int
        """
        rac = random.random()
        if rac <= self.policy[0]:
            return 0
        pol = list(self.policy)
        for i in range(1, len(pol), 1):
            pol[i] += pol[i-1]
            if pol[i] >= rac:
                return i
        return len(pol)


class Provider:
    def __init__(self):
        """
        The init function.
        :return:
        """
        self.queue = []

    def cost(self):
        """
        Returns the total cost of the current queue
        :return: the sum of the current stack
        :rtype: int
        """
        return sum(self.queue)

    def assign(self, task):
        """
        Assigns a task to this provider. Returns the new, total cost.
        :param task: the task (more precisely, its cost)
        :return: the new total cost
        :rtype: int
        """
        self.queue.append(task)
        return self.cost()