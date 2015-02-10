import numpy as np


def print_sim_stats(domain, method, options):
    print(
        '-------------------------------------------------------------------')
    domain_info = ' '
    for arg in domain.__init__.func_code.co_varnames[
            :domain.__init__.func_code.co_argcount]:
        if arg != 'self':
            domain_info += '(' + arg + ',' + repr(getattr(domain, arg)) + '), '
    print('Domain: ' + repr(domain.__class__.__name__) + domain_info[:-2])
    print(
        'Method: ' +
        'Function = ' +
        repr(
            method.f.func_name) +
        ', Projection = ' +
        repr(
            method.Proj.P.func_name))
    params = 'Convergence Criteria: MaxIter: ' + \
        repr(options.Term.Tols[0]) + ', '
    for tol in options.Term.Tols[1:][0]:
        params += repr(tol[0].func_name) + ': ' + repr(tol[1]) + ', '
    print(params[:-2])
    print(
        '-------------------------------------------------------------------')


def print_sim_results(Options, Results, Method, Time):
    print(
        '-------------------------------------------------------------------')
    print('CPU Time: ' + repr(Time))
    for req in Options.repo.perm_requests:
        req_str = req
        if hasattr(req, 'func_name'):
            req_str = req.func_name
        if req in Results.temp_storage:
            print(repr(req_str) + ': ' + repr(Results.temp_storage[req][-1]))
        else:
            print(repr(req_str) + ': ' + repr(Results.perm_storage[req][-1]))
    print('Steps: ' + repr(Results.this_perm_index))
    # print('FEvals: '+`np.sum(Results.PermStorage['Function Evaluations'])`)
    # print('NPs: '+`Method.Proj.NP`)
    # print('Min |X*|: '+`min(abs(Results.temp_storage['Data'][-1]))`)
    # print('Max |X*|: '+`max(abs(Results.temp_storage['Data'][-1]))`)
    print(
        '-------------------------------------------------------------------')
