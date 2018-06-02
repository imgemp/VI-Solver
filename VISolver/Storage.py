try:
    import progressbar  # pip install progressbar2
    bar = progressbar.ProgressBar(max_value=progressbar.UnknownLength)
except ImportError:
    bar = None


class Storage(object):

    def __init__(self,Start,Domain,Method,Options):
        self.thisTempIndex = 0
        self.maxTempIndex = Method.StorageSize
        self.thisPermIndex = 0

        self.Interval = Options.Repo.Interval

        self.TempStorage = Method.InitTempStorage(Start,Domain,Options)

        self.PermStorage = {}
        for req in Options.Repo.PermRequests:
            if req in Method.TempStorage:
                PermItem = Method.TempStorage[req][-1]
            else:
                PermItem = req(Start)
            self.PermStorage[req] = [PermItem]

        self.Timer = Options.Misc.Timer

    def BookKeeping(self,TempStorage):

        # Retrieve New Data
        NewData = TempStorage['Data'][-1]

        # Update TempStorage
        self.TempStorage = TempStorage

        # Update PermStorage
        self.thisPermIndex += 1
        for req in self.PermStorage:
            if self.thisPermIndex % self.Interval == 0:
                if req in self.TempStorage:
                    PermItem = self.TempStorage[req][-1]
                else:
                    PermItem = req(NewData)
                self.PermStorage[req].append(PermItem)

        # Update Progress Bar
        if bar and self.Timer:
            bar.update(self.thisPermIndex)
