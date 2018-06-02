
class Initialization(object):

    def __init__(self,Step=0):
        self.Step = Step


class Termination(object):

    def __init__(self,MaxIter=1,Tols=[],verbose=True):
        self.Tols = [MaxIter]
        self.Tols.append(Tols)
        self.verbose = verbose

    def CheckTols(self,PermRequests,TempRequests):
        for tol in self.Tols[1]:
            if (tol[0] not in PermRequests) and (tol[0] not in TempRequests):
                self.Tols[1].remove(tol)
                print(tol[0].__func__.__name__, 'cannot be used as a',
                      'terminal condition because it is not tracked',
                      'during the descent. Add it to the ``Requests'' list',
                      'for ``Reporting''')
        return self.Tols

    def IsTerminal(self,Record):
        if Record.thisPermIndex >= self.Tols[0]:
            if self.verbose:
                print('\nIteration limit met.')
            return True
        for tol in self.Tols[1]:
            tol_str = tol[0].__func__.__name__
            if tol[0] in Record.TempStorage:
                if Record.TempStorage[tol[0]][-1] <= tol[1]:
                    if self.verbose:
                        print('\n'+tol_str+' condition met.')
                    return True
            elif Record.PermStorage[tol[0]][Record.thisPermIndex] <= tol[1]:
                if self.verbose:
                    print('\n'+tol_str+' condition met.')
                return True
        return False


class Reporting(object):

    def __init__(self,Requests=[],Interval=1):
        self.PermRequests = Requests
        self.Interval = Interval

    def CheckRequests(self,Method,Domain):
        for req in self.PermRequests:
            inTempStorage = (req in Method.TempStorage)
            inDomainFunctions = False
            req_str = req[0]
            if hasattr(req,'__self__'):
                inDomainFunctions = (req.self == Domain)
                req_str = req[0].func_name
            if not (inTempStorage or inDomainFunctions):
                self.PermRequests.remove(req)
                print(repr(req_str), 'cannot be used as a',
                      'terminal condition because it is not tracked',
                      'during the descent.')


class Miscellaneous(object):

    def __init__(self,Min=None,Timer=True):
        self.Min = Min
        self.Timer = Timer


class DescentOptions(object):

    def __init__(self, Init=Initialization(), Term=Termination(),
                 Repo=Reporting(), Misc=Miscellaneous()):
        self.Init = Init
        self.Term = Term
        self.Repo = Repo
        self.Misc = Misc

    def CheckOptions(self,Method,Domain):
        if self.Misc.Min is not None:
            Domain.Min = self.Misc.Min
        # check if requests are either tracked in tempstorage
        # or are available as domain functions
        self.Term.Tols = self.Term.CheckTols(self.Repo.PermRequests,
                                             Method.TempStorage.keys())
