import numpy as np

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

def PrintSimResults(Results,Method,Time):

    print('-------------------------------------------------------------------')
    print('CPU Time: '+`Time`)
    for req in Results.Report:
        print(`req.func_name`+': '+`Results.Report[req][-1]`)
    print('Steps: '+`Results.FEvals.shape[0]-1`)
    print('FEvals: '+`np.sum(Results.FEvals)`)
    print('NPs: '+`Method.Proj.NP`)
    print('Min |X*|: '+`min(abs(Results.Data[-1]))`)
    print('Max |X*|: '+`max(abs(Results.Data[-1]))`)
    print('-------------------------------------------------------------------')