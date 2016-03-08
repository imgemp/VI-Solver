import numpy as np

from VISolver.Domain import Domain


class MixtureMean(Domain):

    def __init__(self,Data,keepData=False,Dim=2):
        self.Data = self.load_data(Data)
        self.keepData = keepData
        self.Dim = Dim

    def load_data(self,Data):
        self.mask = (Data != 0).toarray()
        globalmean = Data.sum()/Data.nnz
        usermean = np.asarray(Data.sum(axis=1)).squeeze()/Data.getnnz(axis=1)
        usermean[np.isnan(usermean)] = globalmean
        moviemean = np.asarray(Data.sum(axis=0)).squeeze()/Data.getnnz(axis=0)
        moviemean[np.isnan(moviemean)] = globalmean
        self.usermean = np.expand_dims(usermean,axis=1)
        self.moviemean = np.expand_dims(moviemean,axis=0)
        return Data

    def predict(self,parameters):
        mu_user = parameters[0]
        # mu_movie = parameters[1]
        mu_movie = 1. - mu_user
        return mu_user*self.usermean + mu_movie*self.moviemean

    def rmse(self,pred,test,mask):
        sqerr = mask*np.asarray(pred - test)**2.
        return np.sqrt(sqerr.sum()/test.nnz)

    def F(self,parameters):
        pred = self.predict(parameters)
        rmse = self.rmse(pred,self.Data,self.mask)
        diff = np.asarray(pred - self.Data)
        dmu_user = np.sum((diff.T*self.usermean.squeeze()).T)
        dmu_movie = np.sum(diff*self.moviemean.squeeze())
        # grad = np.array([dmu_user,dmu_movie])/(rmse*self.Data.nnz)
        grad = np.array([dmu_user-dmu_movie])/(rmse*self.Data.nnz)
        return grad
