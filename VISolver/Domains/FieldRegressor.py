import numpy as np
import sympy
from itertools import combinations

from VISolver.Domain import Domain


class FieldRegressor(Domain):

    def __init__(self,dataset,deg=2):
        # Extract training features and labels from dataset
        self.dataset = dataset
        train_x, train_y = dataset
        self.train_x = train_x
        self.train_y = train_y
        self.N = len(train_y)

        self.deg = deg

        # This function initializes a number of class variables - see below
        self.constructField(train_x.shape[1],deg)

        self.Dim = len(self.c_ij)

    def f(self,Data):
        # Compute the MSE for every labeled pair in training set
        MSE = 0
        for pair,dist in self.train_y:
            # Retrieve training pair
            x1, x2 = self.train_x[pair[0]], self.train_x[pair[1]]

            # Compute starting points and deltas for path integral evaluation
            xo_vals = x1.copy()
            dx_vals = x2 - x1

            # Construct evaluation point (Data = field coefficients, t=xo+dx*t)
            eval_pt = np.hstack((Data,dx_vals,xo_vals))

            # Evaluate path integral (PI contains polynomial degrees)
            dist_pred = self.poly(eval_pt,*self.PI_eval)

            # Increment MSE
            MSE += (dist_pred-dist)**2.

        # Return the average
        return MSE/self.N

    def F(self,Data):
        # Compute MSE gradient wrt field coefficients
        gradMSE = np.zeros(len(self.PIG_eval))
        for pair,dist in self.train_y:
            # Retrieve training pair
            x1, x2 = self.train_x[pair[0]], self.train_x[pair[1]]

            # Compute starting points and deltas for path integral evaluation
            xo_vals = x1.copy()
            dx_vals = x2 - x1

            # Construct evaluation point (Data = field coefficients, t=xo+dx*t)
            eval_pt = np.hstack((Data,dx_vals,xo_vals))

            # Evaluate path integral (PI contains polynomial degrees)
            dist_pred = self.poly(eval_pt,*self.PI_eval)

            # grad_c_ij{MSE} = 2(dist_pred-dist)*d/d_cij{dist_pred}
            #                = 2*err*dfdc
            err = dist_pred - dist
            for idx in xrange(len(gradMSE)):
                dfdc = self.poly(eval_pt,*self.PIG_eval[idx])
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
            tuples += self.fixed_sum_digits(dim,d)[::-1]
        return tuples

    # Evaluate a polynomial given the variables (x), coefficients (c_ij),
    # and monomial degrees (pdl)
    def poly(self,x,c_ij,pdl):
        return np.sum(c_ij*np.product(x**np.asarray(pdl),axis=-1))

    def constructField(self,dim,deg):
        # Retrieve polynomial degree list
        # i.e. all possible monomials with degree <= deg (ignoring coefficients)
        self.pdl = pdl = self.poly_deg_list(dim,deg)

        # Initialize sympy symbolic variables for coefficients
        c_ij = []
        for d in xrange(dim):
            symbol = 'c_' + repr(d)
            c_ij += [sympy.symarray(symbol,len(self.pdl))]
        c_ij = np.asarray(c_ij)

        # Initialize symbolic arrays for features (x_i) and linear paths
        # x_i = xo_i + dx_i*t where xo_i is the starting point for feature i
        # and dx_i is the difference between ending point and starting point
        # for feature i
        x = sympy.symarray('x',dim)
        xo = sympy.symarray('xo',dim)
        dx = sympy.symarray('dx',dim)
        t = sympy.symbols('t')

        # Construct the symbolic field representation
        Field = []
        for d in xrange(dim):
            Field += [self.poly(x,c_ij[d],self.pdl)]

        # Construct the integrand for the path integral (i.e. F dot dr)
        # as a function of the single variable t
        pid = 0
        for i,field in enumerate(Field):
            pid += field.subs(zip(x,xo+dx*t))*dx[i]
        Path_Integrand = sympy.Poly(pid)

        # Evaluate the path integral from 0 to 1
        pil = Path_Integrand.integrate('t')
        Path_Integral = pil.subs('t',1)

        # Compute the gradient of the integral wrt the coefficients for learning
        Grad = []
        for c in c_ij.flatten():
            Grad += [Path_Integral.diff(c)]

        # Store symbolic variables for convenience (not used in computation)
        self.c_ij = c_ij.ravel()
        self.x = x
        self.xo = xo
        self.dx = dx
        self.t = t
        self.pi_symstr = sorted(Path_Integral.free_symbols,key=lambda s: str(s))

        # Store symbolic representations for convenience (not used)
        self.Field = Field
        self.Path_Integrand = Path_Integrand
        self.Path_Integral = Path_Integral
        self.Grad = Grad

        # Convert symbolic path integral to efficient form for fast evaluation
        coeffs = []
        pdl = []
        for p,c in Path_Integral.terms():
            pdl += [tuple([int(pi) for pi in p])]
            coeffs += [float(c)]
        self.PI_eval = (coeffs,pdl)

        # Convert symbolic gradient to efficient form for fast evaluation
        self.PIG_eval = []
        for g in Grad:
            coeffs = []
            pdl = []
            for p,c in g.terms():
                pdl += [tuple([int(pi) for pi in p])]
                coeffs += [float(c)]
            self.PIG_eval += [(coeffs,pdl)]


def constructRandomDataset(N,dim):
    # Construct N, dim-dimensional samples
    train_x = np.random.rand(N,dim)
    # Assign random scalar (e.g. distance) to all sample pairs
    train_y = []
    for pair in combinations(xrange(N),2):
        train_y += [(pair,-5.+10.*np.random.rand())]
    return (train_x, train_y)


def constructSampleDataset(conservative=False,ex=1):
    if conservative:
        # distances are path independent
        train_x = np.array([[0,0],[0,1],[1,0]])
        train_y = [((0,1),5),((0,2),10),((1,2),5)]
    else:
        # distances are path dependent
        if ex == 1:
            # same as conservative example, but with one distance skewed
            train_x = np.array([[0,0],[0,1],[1,0]])
            train_y = [((0,1),5),((0,2),10),((1,2),2.5)]
        elif ex == 2:
            # 4 points around circle
            # opposites have zero distance
            # subsequents have distance of 1 going counter-clockwise
            train_x = np.array([[1,0],[0,1],[-1,0],[0,-1]])
            train_y = [((0,2),0),((1,3),0),
                       ((0,1),1),((1,2),1),((2,3),1),((3,0),1)]
        else:
            # compare to rock-paper-scissors game
            theta = 1j*np.pi*np.asarray([-5./6.,-1./6.,3./6.])
            loc = np.exp(theta)
            train_x = np.vstack((np.real(loc),np.imag(loc))).T
            train_y = [((0,1),1),((1,2),1),((2,0),1)]
    return (train_x, train_y)
