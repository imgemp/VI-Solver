import numpy as np
import sympy

from VISolver.Domain import Domain


class PolyRegressor(Domain):

    def __init__(self,dataset,deg=2):
        # Extract training features and labels from dataset
        self.dataset = dataset
        train_x, train_y = dataset
        self.train_x = train_x
        self.train_y = train_y
        self.N = len(train_y)

        self.deg = deg

        # This function initializes a number of class variables - see below
        self.constructPoly(train_x.shape[1],deg)

        self.Dim = len(self.c_ij)

    def f(self,Data):
        # Compute the MSE for every labeled pair in training set
        MSE = 0
        for pair,dist in self.train_y:
            # Retrieve training pair
            x1, x2 = self.train_x[pair[0]], self.train_x[pair[1]]

            # Construct evaluation points (Data = field coefficients)
            eval_pt_1 = np.hstack((Data,x1))
            eval_pt_2 = np.hstack((Data,x2))

            # Compute predictions and take difference for pairwise distance
            y_pred_1 = self.poly(eval_pt_1,*self.PR_eval)
            y_pred_2 = self.poly(eval_pt_2,*self.PR_eval)
            dist_pred = y_pred_2 - y_pred_1

            # Increment MSE
            MSE += (dist_pred-dist)**2.

        # Return the average
        return MSE/self.N

    def F(self,Data):
        # Compute MSE gradient wrt field coefficients
        gradMSE = np.zeros(len(self.PRG_eval))
        for pair,dist in self.train_y:
            # Retrieve training pair
            x1, x2 = self.train_x[pair[0]], self.train_x[pair[1]]

            # Construct evaluation points (Data = field coefficients)
            eval_pt_1 = np.hstack((Data,x1))
            eval_pt_2 = np.hstack((Data,x2))

            # Compute predictions and take difference for pairwise distance
            y_pred_1 = self.poly(eval_pt_1,*self.PR_eval)
            y_pred_2 = self.poly(eval_pt_2,*self.PR_eval)
            dist_pred = y_pred_2 - y_pred_1

            # grad_c_ij{MSE} = 2(dist_pred-dist)*d/d_cij{dist_pred}
            #                = 2*err*(d_cij{y_pred_2}-d_cij{y_pred_1})
            #                = 2*err*dfdc
            err = dist_pred - dist
            for idx in xrange(len(gradMSE)):
                dfdc_1 = self.poly(eval_pt_1,*self.PRG_eval[idx])
                dfdc_2 = self.poly(eval_pt_2,*self.PRG_eval[idx])
                dfdc = dfdc_2 - dfdc_1
                gradMSE[idx] += 2.*err*dfdc

        # Return the average
        return gradMSE/self.N

    # Following 2 functions adapted from
    # http://stackoverflow.com/questions/8617455/
    # a-nice-way-to-find-all-combinations-that-give-a-sum-of-n
    def iter_fun(self,sum,deepness,myTuple,Total,tuples):
        if deepness == 0:
            if sum == Total:
                tuples += [myTuple]
        else:
            for i in xrange(min(10, Total - sum + 1)):
                self.iter_fun(sum + i,deepness - 1,myTuple + (i,),Total,tuples)

    # See previous comment above
    def fixed_sum_digits(self,digits, Tot):
        tuples = list()
        self.iter_fun(0,digits,tuple(),Tot,tuples)
        return tuples

    # Retrieve list of all monomials of degree <= deg
    def poly_deg_list(self,dim,deg):
        tuples = list()
        for d in xrange(deg+1):
            tuples += self.fixed_sum_digits(dim,d)
        return tuples

    # Evaluate a polynomial given the variables (x), coefficients (c_ij),
    # and monomial degrees (pdl)
    def poly(self,x,c_ij,pdl):
        return np.sum(c_ij*np.product(x**np.asarray(pdl),axis=-1))

    def constructPoly(self,dim,deg):
        # Retrieve polynomial degree list
        # i.e. all possible monomials with degree <= deg (ignoring coefficients)
        self.pdl = self.poly_deg_list(dim,deg)

        # Initialize sympy symbolic variables for coefficients and features
        c_ij = sympy.symarray('c',len(self.pdl))
        x = sympy.symarray('x',dim)

        # Construct polynomial function and gradient
        self.PolyRegressor = sympy.Poly(self.poly(x,c_ij,self.pdl))
        self.Grad = np.asarray([self.PolyRegressor.diff(c) for c in c_ij])

        # Store symbolic variables for convenience (not used in computation)
        self.c_ij = c_ij.ravel()
        self.x = x

        # Convert symbolic polynomial to efficient form for fast evaluation
        coeffs = []
        pdl = []
        for p,c in self.PolyRegressor.terms():
            pdl += [tuple([int(pi) for pi in p])]
            coeffs += [float(c)]
        self.PR_eval = (coeffs,pdl)

        # Convert symbolic gradient to efficient form for fast evaluation
        self.PRG_eval = []
        for g in self.Grad:
            coeffs = []
            pdl = []
            for p,c in g.terms():
                pdl += [tuple([int(pi) for pi in p])]
                coeffs += [float(c)]
            self.PRG_eval += [(coeffs,pdl)]


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
