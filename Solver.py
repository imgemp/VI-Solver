
from Storage import Storage

class Solver(object):

    def __init__(self):

        self.TempStorage = {'Data': ['?']}

    def InitTempStorage(self,Start,Domain,Options):

        print('Every method must define InitTempStorage(self,Start,Domain,Options)' + \
            'to return a dictionary of information the method will need for later use' + \
            'plus the work necessary to obtain that information.')

        self.TempStorage['Data'][-1] = Start

        return self.TempStorage

    def BookKeeping(self,TempData):

        for item in self.TempStorage:
            self.TempStorage[item].pop(0)
            self.TempStorage[item].append(TempData[item])

    def Update(self,Record):

        print('Every method must define Update(self,Record) to return' + \
            'a data packet containing any information required for subsequent' + \
            'iterations by the method as well as the work done to obtain that information.')

        TempData = Record.TempStorage
        self.BookKeeping(TempData)

        return self.TempStorage

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

    return Record





