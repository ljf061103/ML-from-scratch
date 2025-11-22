#Let's solve multiple linear regression(让我们来解决多元线性回归)
import numpy as np
class LinearRegression:

    def __init__(self):
        self.w=None
        self.b=0
        self.my_mean=None
        self.my_std=None
        self.bool_use=False

    def DataProcessing(self,x,bool_train=True):
        data_np=np.array(x)
        if bool_train:            
            self.my_mean=np.mean(data_np,axis=0)
            self.my_std=np.std(data_np,axis=0)
            return (data_np-self.my_mean)/self.my_std
        else:
            return data_np

    def train(self,x,y,learning_rate=0.01,epochs=10000,use=True):
        m=len(x)
        n=len(y)
        if m!=n or m==0:
            print("error")
        else:
            if use:
                x=self.DataProcessing(x)
                self.bool_use=True
            k=len(x[0])
            self.w=np.array([0.0]*k)
            np_x=np.array(x)
            np_y=np.array(y)
            for i in range(epochs):
                dw=np.array([0.0]*k)
                db=0
                y_pre=(self.w*np_x).sum(axis=1)+self.b
                cost=y_pre-np_y
                dw=(cost*np_x.T).sum(axis=1)
                db=cost.sum()
                self.w=self.w-learning_rate*dw/m
                self.b=self.b-learning_rate*db/m

    def pre(self,x):
        if self.bool_use:
            x=self.DataProcessing(x,bool_train=False)
        return (self.w*x).sum(axis=1)+self.b
