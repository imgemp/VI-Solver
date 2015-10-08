import numpy as np
import sympy
from itertools import combinations

from VISolver.Domain import Domain


class FieldRegressor(Domain):

    def __init__(self,dataset,deg=2):
        self.dataset = dataset
        train_x, train_y = dataset
        self.train_x = train_x
        self.train_y = train_y
        self.N = len(train_y)

        self.deg = deg

        self.constructField(train_x.shape[1],deg)

        self.Dim = len(self.c_ij)

    def f(self,Data):
        D_poly_fixed = self.D_poly.subs(zip(self.c_ij,Data))

        MSE = 0
        for pair in combinations(xrange(self.N),2):
            x1, x2 = self.train_x[pair[0]], self.train_x[pair[1]]
            y1, y2 = self.train_y[pair[0]], self.train_y[pair[1]]

            xo_vals = x1.copy()
            dx_vals = x2 - x1

            dy_vals = y2 - y1

            eval_pt = zip(self.xo,xo_vals)+zip(self.dx,dx_vals)
            dy_pred = float(D_poly_fixed.subs(eval_pt))

            MSE += (dy_pred-dy_vals)**2.

        return np.sqrt(MSE)/self.N

    def F(self,Data):
        D_poly_fixed = self.D_poly.subs(zip(self.c_ij,Data))
        grad_fixed = [g.subs(zip(self.c_ij,Data)) for g in self.grad]
        grad_eval = np.zeros(len(self.grad))

        for pair in combinations(xrange(self.N),2):
            x1, x2 = self.train_x[pair[0]], self.train_x[pair[1]]
            y1, y2 = self.train_y[pair[0]], self.train_y[pair[1]]

            xo_vals = x1.copy()
            dx_vals = x2 - x1

            dy_vals = y2 - y1

            eval_pt = zip(self.xo,xo_vals)+zip(self.dx,dx_vals)

            dy_pred = float(D_poly_fixed.subs(eval_pt))

            err = dy_pred - dy_vals

            for idx,g in enumerate(grad_fixed):
                dfdc = g.subs(eval_pt)
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

    def constructField(self,dim,deg):
        pdl = self.poly_deg_list(dim,deg)

        c_ij = []
        for d in xrange(dim):
            symbol = 'c_' + repr(d)
            c_ij += [sympy.symarray(symbol,len(pdl))]
        c_ij = np.asarray(c_ij)

        x = sympy.symarray('x',dim)
        xo = sympy.symarray('xo',dim)
        dx = sympy.symarray('dx',dim)
        t = sympy.symbols('t')

        Field = []
        for d in xrange(dim):
            Field += [self.poly(x,c_ij[d],pdl)]

        dt = 0
        for i,field in enumerate(Field):
            dt += field.subs(zip(x,xo+dx*t))*dx[i]
        dt_poly = sympy.Poly(dt)

        Dt_poly = dt_poly.integrate('t')
        D_poly = Dt_poly.subs('t',1)

        grad = []
        for c in c_ij.flatten():
            grad += [D_poly.diff(c)]

        self.c_ij = c_ij.ravel()
        self.x = x
        self.xo = xo
        self.dx = dx
        self.t = t

        self.Field = Field
        self.dt_poly = dt_poly
        self.pdl = pdl

        self.D_poly = D_poly
        self.grad = grad
