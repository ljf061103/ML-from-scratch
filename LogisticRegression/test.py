from ucimlrepo import fetch_ucirepo 
from logistic_regression import LogisticRegression
import pandas as pd
import time
start=time.time()
model=LogisticRegression()
x_train=[]
y_train=[] 
y_true=[]
x_pre=[]   
breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17) 
X = breast_cancer_wisconsin_diagnostic.data.features 
y = breast_cancer_wisconsin_diagnostic.data.targets 
x_full=X.values.tolist()  
y_full=y['Diagnosis'].map({'M': 1, 'B': 0}).tolist()
x_train=x_full[:450]
y_train=y_full[:450]
print("训练中...")
x_new_train=model.CreatDegree(x_train,degree=2)
print("进度:20%")
model.Log_train(x_new_train,y_train,regularization=True,learning_rate=0.001,use=True,lamb=3)
print("进度:70%")
x_pre=x_full[450:]
y_true=y_full[450:]
x_new_pre=model.CreatDegree(x_pre,degree=2)
print("进度:90%")
y_pre=model.pre(x_new_pre,probability=False,std=0.695)
print("进度:100%")
print("预测值:",y_pre)
print("真实值:",y_true)
end=time.time()
print("用时:",end-start,"秒")