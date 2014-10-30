import numpy as np

class Storage:

    def __init__(self,Start,Domain,Method,Options):
        self.thisTempIndex = 0
        self.maxTempIndex = Method.StorageSize
        self.thisPermIndex = 0

        self.TempStorage = Method.InitTempStorage(Start,Domain,Options)
        
        self.PermStorage = {}
        for req in Options.Repo.PermRequests:
            if req in Method.TempStorage:
                PermItem = Method.TempStorage[req][-1]
            else:
                PermItem = req(Start)
            self.PermStorage[req] = [PermItem]

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
            self.PermStorage[req].append(PermItem)



