import time
import numpy as np

from VISolver.Domains.RegLR import RegularizedLogisticRegression

from VISolver.Solvers.Euler import Euler

from VISolver.Projection import BoxProjection
from VISolver.Solver import Solve
from VISolver.Options import (
    DescentOptions, Miscellaneous, Reporting, Termination, Initialization)
from VISolver.Log import PrintSimResults, PrintSimStats

from VISolver.Utilities import ListONP2NP

from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split

from matplotlib import pyplot as plt

#TRY THREE SPLITS, TRAIN_1 FOR WEIGHTS, 2&3 FOR GENERALIZATION
#TRY L2 GENERALIZATION ERROR INSTEAD OF ABSOLUTE
#NEED TO DO A GRIDSEARCH OVER COEFFICIENTS FOR COMPARISON
def Demo():

    #__ADVERSARIALLY_REGULARIZED_LOGISTIC_REGRESSION__###############################

    # Load MNIST
    digits = datasets.load_digits()
    X = digits.data
    X = StandardScaler().fit_transform(X)
    y = digits.target
    y = (y > 4).astype(np.int)  # binary classification
    print(X.shape)
    
    # Train_1, Train_2, Valid Split
    X_train, X_valid, y_train, y_valid = train_test_split(X,y,test_size=0.20)
    X_train_1, X_train, y_train_1, y_train = train_test_split(X_train,y_train,test_size=0.67)
    X_train_2, X_train_3, y_train_2, y_train_3 = train_test_split(X_train,y_train,test_size=0.5)
    train_1 = (X_train_1,y_train_1)
    train_2 = (X_train_2,y_train_2)
    train_3 = (X_train_3,y_train_3)
    valid = (X_valid,y_valid)

    # Define Domain
    Step = 1e-3
    Domain = RegularizedLogisticRegression(train_1,train_2,train_3,valid,eta=Step)

    # Set Method
    lo = np.hstack([-np.inf*np.ones(Domain.Dim+1),0.,0.])
    hi = np.inf*np.ones(Domain.Dim+3)
    P = BoxProjection(lo=lo,hi=hi)
    Method = Euler(Domain=Domain,FixStep=True,P=P)

    # Initialize Starting Point
    Start = np.hstack([np.random.rand(Domain.Dim+1),0.,0.])

    # Set Options
    Init = Initialization(Step=Step)
    Term = Termination(MaxIter=1e5)
    Repo = Reporting(Requests=['Step','Data',Domain.LogLikelihood,Domain.GenDiff])
    Misc = Miscellaneous()
    Options = DescentOptions(Init,Term,Repo,Misc)

    # Print Stats
    PrintSimStats(Domain,Method,Options)

    # Start Solver
    tic = time.time()
    RegLR_Results = Solve(Start,Method,Domain,Options)
    toc = time.time() - tic

    # Print Results
    PrintSimResults(Options,RegLR_Results,Method,toc)

    LL = np.asarray(RegLR_Results.PermStorage[Domain.LogLikelihood])

    skip = len(LL)//1000

    plt.plot(LL[::skip],'-o',lw=2)
    plt.title('Adversarial Regularization')
    plt.xlabel('Iterations')
    plt.ylabel('Conditional Log-Likelihood')
    plt.show()

    GenDiff = np.asarray(RegLR_Results.PermStorage[Domain.GenDiff])

    plt.plot(GenDiff[::skip],'-o',lw=2)
    plt.title('Adversarial Regularization')
    plt.xlabel('Iterations')
    plt.ylabel('Absolute Generalization Difference')
    plt.show()

    data = ListONP2NP(np.asarray(RegLR_Results.PermStorage['Data']))

    plt.plot(data[::skip,-2],'-o',lw=2,label='L2')
    plt.plot(data[::skip,-1],'-',lw=2,label='L1')
    plt.title('Adversarial Regularization')
    plt.xlabel('Iterations')
    plt.ylabel('Regularization Coefficients')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    Demo()
