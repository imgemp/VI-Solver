from __future__ import print_function

def PrintSimStats(Domain,Method,Options):
    
    print('-------------------------------------------------------------------')
    domain_info = ' '
    for arg in Domain.__init__.func_code.co_varnames[:Domain.__init__.func_code.co_argcount]:
        if arg != 'self': domain_info += '('+arg+','+`getattr(Domain,arg)`+'), '
    print('Domain: '+`Domain.__class__.__name__`+domain_info[:-2])
    print('Method: '+'Function = '+`Method.F.func_name`+', Projection = '+`Method.Proj.P.func_name`)
    params = 'Convergence Criteria: MaxIter: '+`Options.Term.Tols[0]`+', '
    for tol in Options.Term.Tols[1:][0]:
        params += `tol[0].func_name`+': '+`tol[1]`+', '
    print(params[:-2])
    print('-------------------------------------------------------------------')

def PrintSimResults(Options,Results,Method,Time):

    print('-------------------------------------------------------------------')
    print('CPU Time: '+`Time`)
    for req in Options.Repo.PermRequests:
        req_str = req
        if hasattr(req,'func_name'): req_str = req.func_name
        if req in Results.TempStorage:
            print(`req_str`+': '+`Results.TempStorage[req][-1]`)
        else:
            print(`req_str`+': '+`Results.PermStorage[req][-1]`)
    print('Steps: '+`Results.thisPermIndex`)
    # print('FEvals: '+`np.sum(Results.PermStorage['Function Evaluations'])`)
    # print('NPs: '+`Method.Proj.NP`)
    print('Min |X*|: '+`min(abs(Results.TempStorage['Data'][-1]))`)
    print('Max |X*|: '+`max(abs(Results.TempStorage['Data'][-1]))`)
    print('-------------------------------------------------------------------')