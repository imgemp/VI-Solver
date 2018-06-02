import numpy as np

from VISolver.Domain import Domain


class RegularizedLogisticRegression(Domain):

    def __init__(self,train_1,train_2,train_3,valid,eta=1e-2):
        self.train_1 = train_1
        self.train_2 = train_2
        self.train_3 = train_3
        self.valid = valid

        self.X_train_1 = train_1[0]
        self.y_train_1 = train_1[1]
        self.X_train_2 = train_2[0]
        self.y_train_2 = train_2[1]
        self.X_train_3 = train_3[0]
        self.y_train_3 = train_3[1]
        self.X_valid = valid[0]
        self.y_valid = valid[1]

        self.Dim = self.X_train_1.shape[1]
        self.eta = eta

    def LogLikelihood(self,Data,X=None,y=None):
        if X is None or y is None:
            X = self.X_valid
            y = self.y_valid
        wi,w0,c_l2,c_l1 = self.UnpackData(Data)
        py_x = self.Py_x(X,wi,w0,y)
        return np.sum(np.log(py_x))

    def GenDiff(self,Data):
        train_2_ll = self.LogLikelihood(Data,self.X_train_2,self.y_train_2)
        train_3_ll = self.LogLikelihood(Data,self.X_train_3,self.y_train_3)
        return np.abs(train_2_ll-train_3_ll)

    def F(self,Data):
        wi,w0,c_l2,c_l1 = self.UnpackData(Data)

        py_x = self.Py_x(self.X_train_1,wi,w0,y=1)
        dwi = self.X_train_1.T.dot(self.y_train_1-py_x) - c_l2*wi - c_l1*np.sign(wi)
        dw0 = np.sum(self.y_train_1-py_x) - c_l2*w0 - c_l1*np.sign(w0)

        _wi = wi + self.eta*dwi
        _w0 = w0 + self.eta*dw0

        _py_x_2 = self.Py_x(self.X_train_2,_wi,_w0,y=1)
        _dwi_2 = self.X_train_2.T.dot(self.y_train_2-_py_x_2) - c_l2*_wi - c_l1*np.sign(_wi)
        _dw0_2 = np.sum(self.y_train_2-_py_x_2) - c_l2*_w0 - c_l1*np.sign(_w0)

        _py_x_3 = self.Py_x(self.X_train_3,_wi,_w0,y=1)
        _dwi_3 = self.X_train_3.T.dot(self.y_train_3-_py_x_3) - c_l2*_wi - c_l1*np.sign(_wi)
        _dw0_3 = np.sum(self.y_train_3-_py_x_3) - c_l2*_w0 - c_l1*np.sign(_w0)

        _Data = np.hstack([_wi,[_w0,c_l2,c_l1]])
        train_2_ll = self.LogLikelihood(_Data,self.X_train_2,self.y_train_2)
        train_3_ll = self.LogLikelihood(_Data,self.X_train_3,self.y_train_3)
        # gen_subdiff = np.sign(train_2_ll-train_3_ll)
        gen_subdiff = train_2_ll-train_3_ll
        dc_l2 = 0.01*self.eta*gen_subdiff*(wi.dot(_dwi_2-_dwi_3)+w0*(_dw0_2-_dw0_3))
        dc_l1 = 0.01*self.eta*gen_subdiff*(np.sign(wi).dot(_dwi_2-_dwi_3)+np.sign(w0)*(_dw0_2-_dw0_3))
        # print(c_l2,dc_l2)
        # print(c_l1,dc_l1)

        return np.hstack([dwi,[dw0,dc_l2,dc_l1]])

    def Py_x(self,X,wi,w0,y=1):
        assert np.all(np.logical_xor(y==1,y==0))
        inner = np.dot(X,wi)
        inner = inner + w0
        py_x = 1/(1+np.exp(np.dot(X,wi)+w0))
        if isinstance(y,int):
            if y==1:
                return 1-py_x
        else:
            return py_x

    def UnpackData(self,Data):
        wi = Data[:-3]
        w0 = Data[-3]
        c_l2 = Data[-2]
        c_l1 = Data[-1]
        return wi,w0,c_l2,c_l1
