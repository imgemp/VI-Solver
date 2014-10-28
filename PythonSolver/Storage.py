import numpy as np

class Storage:

    def __init__(self,Start,Domain,Method,Options):
        self.thisTempIndex = 0
        self.maxTempIndex = len(Method.TempStorage.items()[0][1])
        self.thisPermIndex = 0
        self.maxPermIndex = Options.Term.Tols[0]+1

        self.TempStorage = Method.InitTempStorage(Start,Domain,Options)
        
        self.PermStorage = {}
        for req in Options.Repo.PermRequests:
            if req in Method.TempStorage:
                PermItem = Method.TempStorage[req][-1]
            else:
                PermItem = req(Start)
            if isinstance(PermItem,np.ndarray):
                self.PermStorage[req] = self.maxPermIndex*[np.reshape([np.NaN for i in PermItem.flatten()],PermItem.shape)]
            else:
                self.PermStorage[req] = self.maxPermIndex*[np.NaN]
            self.PermStorage[req][0] = PermItem

    def BookKeeping(self,TempStorage):

        # Retrieve New Data
        NewData = TempStorage['Data'][-1]

        # Update TempStorage
        self.TempStorage = TempStorage

        # Update PermStorage
        self.thisPermIndex += 1
        for req in self.PermStorage:
            if req in self.TempStorage:
                PermItem = self.TempStorage[req][-1]
            else:
                PermItem = req(NewData)
            np.append(self.PermStorage[req],[PermItem])

    def RemoveUnused(self):
        for req in self.PermStorage:
            self.PermStorage[req] = self.PermStorage[req][:self.thisPermIndex+1]



