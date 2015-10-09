import numpy as np
import sympy

from VISolver.Domain import Domain


class PolyRegressor(Domain):

    def __init__(self,dataset,deg=2):
        self.dataset = dataset
        train_x, train_y = dataset
        self.train_x = train_x
        self.train_y = train_y
        self.N = len(train_y)

        self.deg = deg

        self.constructPoly(train_x.shape[1],deg)

        self.Dim = len(self.c_ij)

    def f(self,Data):
        poly_fixed = self.poly.subs(zip(self.c_ij,Data))

        MSE = 0
        for pair,dist in self.train_y:
            x1, x2 = self.train_x[pair[0]], self.train_x[pair[1]]

            eval_pt_1 = zip(self.x,x1)
            eval_pt_2 = zip(self.x,x2)
            y_pred_1 = float(poly_fixed.subs(eval_pt_1))
            y_pred_2 = float(poly_fixed.subs(eval_pt_2))
            dist_pred = y_pred_2 - y_pred_1

            MSE += (dist_pred-dist)**2.

        return np.sqrt(MSE)/self.N

    def F(self,Data):
        poly_fixed = self.poly.subs(zip(self.c_ij,Data))
        grad_fixed = [g.subs(zip(self.c_ij,Data)) for g in self.dpoly]
        grad_eval = np.zeros(len(self.dpoly))

        for pair,dist in self.train_y:
            x1, x2 = self.train_x[pair[0]], self.train_x[pair[1]]

            eval_pt_1 = zip(self.x,x1)
            eval_pt_2 = zip(self.x,x2)
            y_pred_1 = float(poly_fixed.subs(eval_pt_1))
            y_pred_2 = float(poly_fixed.subs(eval_pt_2))
            dist_pred = y_pred_2 - y_pred_1

            err = dist_pred - dist

            for idx,g in enumerate(grad_fixed):
                dfdc = g.subs(eval_pt_2) - g.subs(eval_pt_1)
                grad_eval[idx] += err*dfdc

        grad_eval *= 2./self.N

        return grad_eval

    def iter_fun(self,sum,deepness,myTuple,Total,tuples):
        if deepness == 0:
            if sum == Total:
                tuples += [myTuple]
        else:
            for i in xrange(min(10, Total - sum + 1)):
                self.iter_fun(sum + i,deepness - 1,myTuple + (i,),Total,tuples)

    def fixed_sum_digits(self,digits, Tot):
        tuples = list()
        self.iter_fun(0,digits,tuple(),Tot,tuples)
        return tuples

    def poly_deg_list(self,dim,deg):
        tuples = list()
        for d in xrange(deg+1):
            tuples += self.fixed_sum_digits(dim,d)
        return tuples

    def poly(self,x,c_ij,pdl):
        return np.sum(c_ij*np.product(x**np.asarray(pdl),axis=1))

    def constructPoly(self,dim,deg):
        pdl = self.poly_deg_list(dim,deg)
        c_ij = sympy.symarray('c',len(pdl))
        x = sympy.symarray('x',dim)

        poly = sympy.Poly(self.poly(x,c_ij,pdl))
        dpoly = np.asarray([poly.diff(c) for c in c_ij])

        self.c_ij = c_ij.ravel()
        self.x = x

        self.poly = poly
        self.dpoly = dpoly
        self.pdl = pdl


def conv2field(c_ij):
    # np.asarray([poly.diff(xi).coeffs() for xi in x])
    # only works for dim=2,deg=2 for now
    c_ij_field = np.empty((2,3))
    c_ij_field[0,0] = c_ij[2]
    c_ij_field[0,1] = c_ij[4]
    c_ij_field[0,2] = 2*c_ij[5]
    c_ij_field[1,0] = c_ij[1]
    c_ij_field[1,1] = 2*c_ij[3]
    c_ij_field[1,2] = c_ij[4]
    return c_ij_field.flatten()
