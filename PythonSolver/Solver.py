import numpy as np

from Storage import *

class Solver(object):

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

    #Create Storage Object for Record Keeping
    Record = Storage(Start,Options)

    #Begin Solving
    while not Options.Term.IsTerminal(Record):

        #Compute New Data Using Update Method
        Data, Step, FEvals = Method.Update(Record,Domain,Step) #should also report projections

        #Record Update Stats
        Record.BookKeeping(Data,Step,FEvals)

    #Remove Unused Entries
    Record.RemoveUnused()

    return Record





