import numpy as np


class Initialization:

    def __init__(self, step=0):
        self.step = step


class Termination:

    def __init__(self, max_iter=1, tols=[]):
        self.tols = [max_iter]
        self.tols.append(tols)

    def check_tols(self, perm_requests, temp_requests):
        for tol in self.tols[1]:
            if (not (tol[0] in perm_requests)) and (
                    not (tol[0] in temp_requests)):
                self.tols.remove(tol)
                print(
                    repr(
                        tol[0].func_name) +
                    ' cannot be used as a terminal condition because it is not tracked during the descent.')
        return self.tols

    def is_terminal(self, record):
        if record.this_perm_index >= self.tols[0]:
            return True
        for tol in self.tols[1]:
            if tol[0] in record.temp_storage:
                if record.temp_storage[tol[0]][-1] <= tol[1]:
                    return True
            else:
                if record.perm_storage[tol[0]][record.this_perm_index].any() <= tol[1]:
                    return True


class Reporting:
    def __init__(self, requests=[]):
        self.perm_requests = requests

    def check_requests(self, method, domain):
        for req in self.perm_requests:
            in_temp_storage = (req in method.temp_storage)
            in_domain_functions = False
            req_str = req[0]
            if hasattr(req, '__self__'):
                in_domain_functions = (req.self == domain)
                req_str = req[0].func_name
            if not (in_temp_storage or in_domain_functions):
                self.perm_requests.remove(req)
                print(
                    repr(req_str) +
                    ' cannot be used as a terminal condition because it is not tracked during the descent.')


class Miscellaneous:
    def __init__(self, min_val=None):
        self.Min = min_val


class DescentOptions:
    def __init__(self, init=Initialization(), term=Termination(), repo=Reporting(), misc=Miscellaneous()):
        self.init = init
        self.term = term
        self.repo = repo
        self.misc = misc

    def check_options(self, method, domain):
        if not self.misc.Min is None:
            domain.Min = self.misc.Min
        # check if requests are either tracked in tempstorage or are available
        # as domain functions
        self.term.tols = self.term.check_tols(self.repo.perm_requests, method.temp_storage.keys())
