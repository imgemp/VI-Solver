import numpy as np

from Storage import *

class Solver(object):

    def Update(self,Record):

        print('Every method must define Update(self,Record,Domain,Step) to return' + \
            'the next iterate, next stepsize, and number of function evaluations used.')

        Data = None
        Step = None
        FEvals = None

        return Data, Step, FEvals

def Solve(Start,Method,Domain,Options):

    #Record Data Dimension 
    Domain.Dim = Start.size # is this necessary?

    #Check Validity of Options
    Options.CheckOptions(Method,Domain)

    #Create Storage Object for Record Keeping
    Record = Storage(Start,Domain,Method,Options)

    #Begin Solving
    while not Options.Term.IsTerminal(Record):

        #Compute New Data Using Update Method
        TempStorage = Method.Update(Record) #should also report projections

        #Record Update Stats
        Record.BookKeeping(TempStorage)

    #Remove Unused Entries
    Record.RemoveUnused()

    return Record





