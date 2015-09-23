<<<<<<< HEAD
=======
# import numpy as np

>>>>>>> master
from VISolver.Domain import Domain


class NewDomain(Domain):

    def __init__(self,F,Dim=1):
        self.Dim = Dim
        assert len(F.func_code.co_varnames) == 1
        self.F_ = F

    def F(self,Data):
        return self.F_(Data)
