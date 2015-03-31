__author__ = 'clemens'

import numpy as np


def decaying_average_estimator(value_histo, averaging_window, window_reps, decaying_rate=.9, grad_history=None):
    """
    This is the decaying average estimator. It computes a weighted average, where the weights are
    decaying backwards.
    :param value_histo:
    :param averaging_window: the number of observations considered to be locally stable
    :param window_reps: the number of observation windows considered to be relevant
    :param decaying_rate: the rate at which the observations decay
    :param grad_history: another value history where the gradients on the observation-window gaps are considered
    :return:
    """
    # indices_grad = np.array([-1*averaging_window*i for i in range(1, window_reps+1)], dtype='int')
    # indices_val = np.array([-1*averaging_window*i-1 for i in range(window_reps)], dtype='int')
    # wlen = len(indices_grad)
    # # if the second parameter, i.e. the gradient-weight, is not specified
    # if grad_history is None:
    #     val_diff = np.ones((window_reps))
    # else:
    #     val_diff = np.array(grad_history)[indices_val] - np.array(grad_history)[indices_val-1]
    # weights = np.array([decaying_rate**(wlen-i) for i in range(wlen)])
    values = np.array(value_histo)
    value_length = len(value_histo)
    weights = np.array([decaying_rate**(value_length-i) for i in range(value_length)])
    return np.dot(values, weights)


def population_based_incremental_learning(data_history, value_history, averaging_window, window_reps,
                                          iterations=1000, size=100):
    # reclassifying the data
    # finding the possible data values:
    hist_num, hist_val = np.histogram(data_history)
    dat = np.zeros((len(data_history), sum(hist_num > 0)))
    val = hist_val[hist_num > 0]
    val_ind = np.digitize(data_history, val)
    bits = len(val_ind)
    prob_dist = np.ones((bits, )) * .5
    best_cost = 1.e300
    for i in range(bits):
        dat[i, val_ind[i]-1] = 1

    # applying the algorithm
    for i in range(iterations):
        genes = np.zeros((size, bits))
        for gene in genes:
            for j in range(bits):
                if np.random.random() < prob_dist[j]:
                    gene[j] = True

        # calculate the costs
        costs = np.zeros((size,))
        for j in range(size):
            costs[j] = 0#compute_cost
    pass






#
# public void optimize() {
#     final int totalBits = getTotalBits(domains);
#     final double[] probVec = new double[totalBits];
#     Arrays.fill(probVec, 0.5);
#     bestCost = POSITIVE_INFINITY;
#
#     for (int i = 0; i < ITER_COUNT; i++) {
#         // Creates N genes
#         final boolean[][] genes = new boolean[N][totalBits];
#         for (boolean[] gene : genes) {
#             for (int k = 0; k < gene.length; k++) {
#                 if (rand.nextDouble() < probVec[k])
#                     gene[k] = true;
#             }
#         }
#
#         // Calculate costs
#         final double[] costs = new double[N];
#         for (int j = 0; j < N; j++) {
#             costs[j] = costFunc.cost(toRealVec(genes[j], domains));
#         }
#
#         // Find min and max cost genes
#         boolean[] minGene = null, maxGene = null;
#         double minCost = POSITIVE_INFINITY, maxCost = NEGATIVE_INFINITY;
#         for (int j = 0; j < N; j++) {
#             double cost = costs[j];
#             if (minCost > cost) {
#                 minCost = cost;
#                 minGene = genes[j];
#             }
#             if (maxCost < cost) {
#                 maxCost = cost;
#                 maxGene = genes[j];
#             }
#         }
#
#         // Compare with the best cost gene
#         if (bestCost > minCost) {
#             bestCost = minCost;
#             bestGene = minGene;
#         }
#
#         // Update the probability vector with max and min cost genes
#         for (int j = 0; j < totalBits; j++) {
#             if (minGene[j] == maxGene[j]) {
#                 probVec[j] = probVec[j] * (1d - learnRate) +
#                         (minGene[j] ? 1d : 0d) * learnRate;
#             } else {
#                 final double learnRate2 = learnRate + negLearnRate;
#                 probVec[j] = probVec[j] * (1d - learnRate2) +
#                         (minGene[j] ? 1d : 0d) * learnRate2;
#             }
#         }
#
#         // Mutation
#         for (int j = 0; j < totalBits; j++) {
#             if (rand.nextDouble() < mutProb) {
#                 probVec[j] = probVec[j] * (1d - mutShift) +
#                         (rand.nextBoolean() ? 1d : 0d) * mutShift;
#             }
#         }
#     }
# }


# def decaying_average_estimator(value_histo, averaging_window, window_reps, decaying_rate=.9, grad_history=None):
#     """
#     This is the decaying average estimator. It computes a weighted average, where the weights are
#     decaying backwards.
#     :param value_histo:
#     :param averaging_window: the number of observations considered to be locally stable
#     :param window_reps: the number of observation windows considered to be relevant
#     :param decaying_rate: the rate at which the observations decay
#     :param grad_history: another value history where the gradients on the observation-window gaps are considered
#     :return:
#     """
    # indices_grad = np.array([-1*averaging_window*i for i in range(1, window_reps+1)], dtype='int')
    # indices_val = np.array([-1*averaging_window*i-1 for i in range(window_reps)], dtype='int')
    # wlen = len(indices_grad)
    # # if the second parameter, i.e. the gradient-weight, is not specified
    # if grad_history is None:
    #     val_diff = np.ones((window_reps))
    # else:
    #     val_diff = np.array(grad_history)[indices_val] - np.array(grad_history)[indices_val-1]
    # weights = np.array([decaying_rate**(wlen-i) for i in range(wlen)])