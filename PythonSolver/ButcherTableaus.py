import numpy as np

#Butcher Tableaus
class ButcherTableau:

    def __init__(self,Table,Order=None,Mem=0,EAS=['Traditional']):
        if type(Table) == type(''):
            if (Table == 'Euler') or (Table == 'Gradient'):
                self.Table = np.array([[1.0]])
                self.Order = 1.0
                self.Mem = 0
                self.EAS = ['Traditional']
            elif Table == 'Extra-Gradient':
                self.Table = np.array([[1.0,0.0],[0.0,1.0]])
                self.Order = 1.0
                self.Mem = 0
                self.EAS = ['Traditional']
            elif Table == 'Heun':
                self.Table = np.array([[1.0,0.0],[0.5,0.5]])
                self.Order = 2.0
                self.Mem = 0
                self.EAS = ['Traditional']
            elif Table == 'Heun-Euler':
                self.Table = np.array([[1.0,0.0],[0.5,0.5],[1.0,0.0]])
                self.Order = 2.0
                self.Mem = 0
                self.EAS = ['Traditional','Runge-Kutta','R-K Local Extrapolation']
            elif Table == 'Cash-Karp':
                self.Table = np.array([[1.0/5.0,0.0,0.0,0.0,0.0,0.0],\
                                    [3.0/40.0,9.0/40.0,0.0,0.0,0.0,0.0],\
                                    [3.0/10.0,-9.0/10.0,6.0/5.0,0.0,0.0,0.0],\
                                    [-11.0/54.0,5.0/2.0,-70.0/27.0,35.0/27.0,0.0,0.0],\
                                    [1631.0/55296.0,175.0/512.0,575.0/13824.0,44275.0/110592.0,253.0/4096.0,0.0],\
                                    [37.0/378.0,0.0,250.0/621.0,125.0/594.0,0.0,512.0/1771.0],\
                                    [2825.0/27648.0,0.0,18575.0/48384.0,13525.0/55296.0,277.0/14336.0,0.25]\
                                    ])
                self.Order = 5.0
                self.Mem = 0
                self.EAS = ['Traditional','Runge-Kutta','R-K Local Extrapolation']
            else:
                print(Table+" is not an option. Please choose from Euler, Prox, Heun, Heun-Euler, or Heun-Euler LE.")
        elif Order == None:
            print("You must specify the order of accuracy implied by the ButcherTableau")
            #how do I destroy this empty instance of ButcherTableau?
        else:
            self.Table = Table
            self.Order = Order
            self.Mem = Mem
            self.EAS = EAS
        self.TableA = None
        self.TableB = None

    def setup(self,AS):
        self.TableA = None
        self.TableB = None