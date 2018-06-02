from __future__ import print_function
import numpy as np


def abbrev_repr(attr):
    if isinstance(attr,tuple):
        repr_attr = tuple()
        for a in attr:
            if isinstance(a,np.ndarray):
                repr_attr += ('np.ndarray of shape '+repr(a.shape),)
            elif isinstance(a,list):
                repr_attr += ('list of length '+repr(len(a)),)
            else:
                repr_attr += (repr(a),)
        return repr_attr
    elif isinstance(attr,np.ndarray):
        return 'np.ndarray of shape '+repr(attr.shape)
    elif isinstance(attr,list):
        return 'list of length '+repr(len(attr))
    else:
        return attr


def PrintSimStats(Domain, Method, Options):
    print('------------------------------------------------------------------')
    domain_info = []
    domain_init_func = Domain.__init__.__func__.__code__
    domain_args = domain_init_func.co_varnames[:domain_init_func.co_argcount]
    domain_info = ['(%s,%r)' % (arg, abbrev_repr(getattr(Domain,arg)))
                   for arg in domain_args if arg != 'self']
    print('Domain: Name = %r, %s' % (Domain.__class__.__name__,', '.join(domain_info)))
    method_info = []
    method_init_func = Method.__init__.__func__.__code__
    method_args = method_init_func.co_varnames[:method_init_func.co_argcount]
    method_info = ['(%s,%r)' % (arg, abbrev_repr(getattr(Method,arg)))
                   for arg in method_args if arg not in ('self','Domain','P')]
    print('Method: Name = %r, Projection = %r, %s' %
          (Method.__class__.__name__,
           Method.Proj.__class__.__name__,', '.join(method_info)))
    params = ['Convergence Criteria: MaxIter: %r' % Options.Term.Tols[0]]
    for tol in Options.Term.Tols[1:][0]:
        if isinstance(tol[1],(str,bool)):
            params.append('%r: %r' % (tol[0].__func__.__name__, tol[1]))
        else:
            params.append('%r: %g' % (tol[0].__func__.__name__, tol[1]))
    print(*params, sep=', ')
    print('------------------------------------------------------------------')


def PrintSimResults(Options, Results, Method, Time):
    print('------------------------------------------------------------------')
    print('CPU Time: %.3f' % Time)
    for req in Options.Repo.PermRequests:
        req_str = req
        if hasattr(req,'__func__'):
            req_str = req.__func__.__name__
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
        steps = Results.PermStorage['Step']
        if isinstance(steps[0],np.ndarray):
            steps = np.mean(steps,axis=1)
        print('Step Sum: %g' % sum(steps))
    print('Min |X*|: %.3f' % np.min(np.abs(Results.TempStorage['Data'][-1])))
    print('Max |X*|: %.3f' % np.max(np.abs(Results.TempStorage['Data'][-1])))
    print('------------------------------------------------------------------')
