import numpy as np

class Domain:

    def __init__(self):
        print('This is a generic domain object.  You need to pick a specific domain to use.')

    def CheckRequests(self,Requests):
        Requests = list(set(Requests))
        for req in Requests:
            if not (req in self.Fun.keys()):
                Requests.remove(req)
                print(req+' is not reported by this domain.')
        return Requests





