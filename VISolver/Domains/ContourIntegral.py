import numpy as np
from VISolver.Domain import Domain


class ContourIntegral(Domain):
    def __init__(self,thisDomain,thisContour):
        # thisContour is a parameterized curve (i.e. f: scalar t --> X in Rn)
        self.contour = thisContour.contour
        self.dcontour = thisContour.dcontour
        self.domain = thisDomain

    def F(self,t):
        Data = self.contour(t)
        return np.atleast_1d(np.dot(self.domain.F(Data),self.dcontour(t)))


class LineContour(object):
    def __init__(self,start,end):
        self.start = start
        self.end = end

    def contour(self,t):
        return self.start + (self.end-self.start)*t

    def dcontour(self,t):
        return self.end-self.start
