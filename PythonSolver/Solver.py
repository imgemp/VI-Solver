import numpy as np

from Projections import *
from Path import *
from Utilities import *

class Solver(object,Projection):

    def Update(self,Record,Domain,Step):

        print('Every method must define Update(self,Record,Domain,Step) to return' + \
            'the next iterate, next stepsize, and number of function evaluations used.')

        Data = None
        Step = None
        FEvals = None

        return Data, Step, FEvals

def Solve(Start,Method,Domain,Options):

    #Record Data Dimension
    Domain.Dim = Start.size

    #Check Validity of Options
    Options.CheckOptions(Method,Domain)
    Step = Options.Init.Step

    #Create Path Object for Record Keeping
    Record = Path(Start,Options)

    #Begin Solving
    while not Options.Term.IsTerminal(Record):

        #Compute New Data Using Update Method
        Data, Step, FEvals = Method.Update(Record,Domain,Step)

        #Record Path Stats
        Record.BookKeeping(Data,Step,FEvals)

    #Remove Unused Entries
    Record.RemoveUnused()

    return Record





