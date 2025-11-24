import numpy as np
from itertools import combinations_with_replacement
class LinearRegression:

    def __init__(self):
        self.w=None
        self.b=0
        self.my_mean=None
        self.my_std=None
        self.bool_use=False
        self.cost_before=float('inf')
        self.cost_last=0

    def CreateDegree(self,x,degree=1):#添加特征，升幂
        if degree==1:
            return x
        else:
            x_new=[]
            x_list=np.array(x)
            x_with,x_hight=x_list.shape
            combos=[]
            for i in range(1,degree+1):
                combos.extend(combinations_with_replacement(range(x_hight),i))
            for j in combos:
                x_new.append(np.prod(x_list[:,j],axis=1))
        return np.column_stack(x_new)

    def DataProcessing(self,x):#数据标准化
        data_np=np.array(x)   
        self.my_mean=np.mean(data_np,axis=0)
        self.my_std=np.std(data_np,axis=0)
        return (data_np-self.my_mean)/self.my_std

    def Regularization_train(self,x,y,learning_rate,epochs,threshold,lamb):#正则化
        m=len(x)
        k=len(x[0])
        self.w=np.array([0.0]*k)
        np_x=np.array(x)
        np_y=np.array(y)
        for i in range(epochs):
            dw=np.array([0.0]*k)
            db=0
            y_pre=(self.w*np_x).sum(axis=1)+self.b
            cost=y_pre-np_y
            self.cost_last=cost
            dw=(cost*np_x.T).sum(axis=1)+(lamb/m)*self.w
            db=cost.sum()
            self.w=self.w-learning_rate*dw/m
            self.b=self.b-learning_rate*db/m
            self.cost_last=(cost**2).sum()/(2*m)+(lamb/(2*m))*np.sum(self.w**2)
            if abs(self.cost_before-self.cost_last)<threshold:
                break
            self.cost_before=self.cost_last

    def train(self,x,y,learning_rate=0.001,epochs=100000,use=True,threshold=1e-6,regularization=False,lamb=1):#模型训练
        m=len(x)
        n=len(y)
        if m!=n or m==0:
            print("error")
        else:
            if use:
                x=self.DataProcessing(x)
                self.bool_use=True
            if regularization:
                self.Regularization_train(x,y,learning_rate,epochs,threshold,lamb)
            else:
                k=len(x[0])
                self.w=np.array([0.0]*k)
                np_x=np.array(x)
                np_y=np.array(y)
                for i in range(epochs):
                    dw=np.array([0.0]*k)
                    db=0
                    y_pre=(self.w*np_x).sum(axis=1)+self.b
                    cost=y_pre-np_y
                    self.cost_last=cost
                    dw=(cost*np_x.T).sum(axis=1)
                    db=cost.sum()
                    self.w=self.w-learning_rate*dw/m
                    self.b=self.b-learning_rate*db/m
                    self.cost_last=(cost**2).sum()/(2*m)
                    if abs(self.cost_before-self.cost_last)<threshold:
                        break
                    self.cost_before=self.cost_last

    def pre(self,x):#预测值
        if self.bool_use:
            x=self.DataProcessing(x)
        else:
            x=np.array(x)
        return (self.w*x).sum(axis=1)+self.b