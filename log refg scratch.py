import numpy as np
class logistic_regression:
    def __init__(self,lr=0.001,n_iters=1000):
        self.lr=lr
        self.n_iters=n_iters
        self.weights=None
        self.biases=None
        
    def fit(self,x,y):
        n_samples,n_features=x.shape
        self.weights=np.zeros(n_features)
        self.biases=0
        for i in range(self.n_iters):
            linear_model=np.dot(x,self.weights)+self.biases
            y_pred=self.sigmoid(linear_model)
            dw=(1/n_samples)*np.dot(x.T ,(y_pred-y))
            db=(1/n_samples)*np.sum(y_pred-y)
            self.weights-=self.lr*dw
            self.biases-=self.lr*db            
            
    def Predict(self,x):
        lin_model=np.dot(x,self.weights)+self.biases
        y_pred=1/(1+np.exp(-lin_model))
        y_pred_cls=[1 if y_pred>0.5 else 0 for i in y_pred]
        return y_pred_cls
                 
    def sigmoid(self,x):
        return 1/(1+np.exp(-x))