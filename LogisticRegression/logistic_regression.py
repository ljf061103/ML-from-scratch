from linear_regression import LinearRegression
import numpy as np
class LogisticRegression:

    def __init__(self):
        self.w=None
        self.b=0
        self.cost=0.0
        self.bool_use=False
        self.lr=LinearRegression()

    def CreatDegree(self,x,degree=1):#添加特征，升幂
        x=self.lr.CreateDegree(x,degree)
        return x
    
    def regular_train(self,x,y,learning_rate,epochs,threshold,lamb):#正则化
        m=len(x[0])
        n=len(x)
        np_x=np.array(x)
        np_y=np.array(y)
        for i in range(epochs):
            dw=np.array([0.0]*m)
            db=0
            y_pre=(self.w*np_x).sum(axis=1)+self.b
            sigmoid=1/(1+np.exp(-y_pre))
            sigmoid = np.clip(sigmoid, 1e-10, 1 - 1e-10)
            self.cost=np.mean(-np_y*np.log(sigmoid)-(1-np_y)*np.log(1-sigmoid))
            lost=sigmoid-np_y
            dw=(lost*np_x.T).sum(axis=1)+(lamb/n)*self.w
            db=lost.sum()
            self.w=self.w-learning_rate*dw/n
            self.b=self.b-learning_rate*db/n
            if self.cost<threshold:
                break

    def Log_train(self,x,y,learning_rate=0.01,epochs=100000,use=True,threshold=1e-6,regularization=False,lamb=1):#模型训练
        m=len(x[0])
        n=len(x)
        k=len(y)
        if n!=k or n==0:
            print("error")
        else:
            self.w=np.array([0.0]*m)
            if use:
                x=self.lr.DataProcessing(x)#数据标准化
                self.bool_use=True
            if regularization:
                self.regular_train(x,y,learning_rate,epochs,threshold,lamb)
            else:
                np_x=np.array(x)
                np_y=np.array(y)
                for i in range(epochs):
                    dw=np.array([0.0]*m)
                    db=0
                    y_pre=(self.w*np_x).sum(axis=1)+self.b
                    sigmoid=1/(1+np.exp(-y_pre))
                    sigmoid = np.clip(sigmoid, 1e-10, 1 - 1e-10)
                    self.cost=np.mean(-np_y*np.log(sigmoid)-(1-np_y)*np.log(1-sigmoid))
                    lost=sigmoid-np_y
                    dw=(lost*np_x.T).sum(axis=1)
                    db=lost.sum()
                    self.w=self.w-learning_rate*dw/n
                    self.b=self.b-learning_rate*db/n
                    if self.cost<threshold:
                        break

    def pre(self,x,probability=True,std=0.5):#预测值
        if self.bool_use:
            x=self.lr.DataProcessing(x)
        else:
            x=np.array(x)
        linear_output=np.dot(x,self.w.T)+self.b
        y_pre=1/(1+np.exp(-linear_output))
        if probability:
            return y_pre
        else:
            return (y_pre>=std).astype(int)