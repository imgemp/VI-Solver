from __future__ import print_function


def PrintSimStats(Domain, Method, Options):
    print('-------------------------------------------------------------------')
    domain_info = []
    domain_init_func = Domain.__init__.func_code
    domain_args = domain_init_func.co_varnames[:domain_init_func.co_argcount]
    domain_info = ['(%s,%r)' % (arg, getattr(Domain,arg))
                   for arg in domain_args if arg != 'self']
    print('Domain: %r %s' % (Domain.__class__.__name__, ', '.join(domain_info)))
    print('Method: Function = %r, Projection = %r' % (Method.F.func_name,
                                                      Method.Proj.P.func_name))
    params = ['Convergence Criteria: MaxIter: %r' % Options.Term.Tols[0]]
    for tol in Options.Term.Tols[1:][0]:
        params.append('%r: %r' % (tol[0].func_name, tol[1]))
    print(*params, sep=', ')
    print('-------------------------------------------------------------------')


def PrintSimResults(Options, Results, Method, Time):
    print('-------------------------------------------------------------------')
    print('CPU Time: %.3f' % Time)
    for req in Options.Repo.PermRequests:
        req_str = req
        if hasattr(req,'func_name'):
            req_str = req.func_name
        if req in Results.TempStorage:
            try:
                print('%s: %g' % (req_str, Results.TempStorage[req][-1]))
            except TypeError:
                print('%s: %s' % (req_str, Results.TempStorage[req][-1]))
        else:
            try:
                print('%s: %g' % (req_str, Results.PermStorage[req][-1]))
            except TypeError:
                print('%s: %s' % (req_str, Results.PermStorage[req][-1]))
    print('Steps: %d' % Results.thisPermIndex)
    if 'Step' in Options.Repo.PermRequests:
        print('Step Length: %g' % sum(Results.TempStorage['Step']))
    print('Min |X*|: %.3f' % min(abs(Results.TempStorage['Data'][-1])))
    print('Max |X*|: %.3f' % max(abs(Results.TempStorage['Data'][-1])))
    print('-------------------------------------------------------------------')
