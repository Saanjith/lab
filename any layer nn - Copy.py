import numpy as np
import copy
class NN:
    def __init__(self,ndim):
        print('Neural Network')
        self.weights = []
        self.ndim = ndim #number of features in input
        self.activations = []
        self.layers = []
        
    def addlayer(self,layer_size,activation=None):
        if self.weights == []:
            self.weights.append(np.random.random((self.ndim+1,layer_size))*0.1-0.05)
        else:
            prev = self.weights[-1].shape[1]
            self.weights.append(np.random.random((prev+1,layer_size))*0.1-0.05)
        
        if activation=='relu':
            self.activations.append(NN.ReLU)
        elif activation=='sig':
            self.activations.append(NN.sig)
        else:
            self.activations.append(lambda x,der=False :np.ones(x.shape) if der else x) #literaly does nothing
            
    def sig(x,der=False):
        if not der:
            return 1/(1+np.exp(-x))
        return x*(1-x)
    
    def ReLU(x,der=False):
        if not der:
            return (x>0)*x
        return x>0
    
    def biaslayer(layer):
        return np.append(np.ones((len(layer),1)),layer,axis=1)
    
    def forward(self,x):
        self.layers = [x]
        layer = x
        for w,act in zip(self.weights,self.activations):
            layer = NN.biaslayer(layer)
            layer = layer@w
            layer = act(layer)
            self.layers.append(layer)
        return layer
    
    def backward(self,y,ycap):
        error = y-ycap #starting error
        for i in range(len(self.weights)-1,-1,-1):
            act = self.activations[i]
            layer = self.layers[i+1]
            w = self.weights[i]
            
            delta = error * act(layer,der=True) #delta calculation
            
            nextlayer = copy.deepcopy(self.layers[i])
            nextlayer = NN.biaslayer(nextlayer)
            w += ( nextlayer.T @ delta )*1/len(y)  #altering weights
            
            error = (delta @ w.T)[:,1:] #error for next layer
        
    def train(self,x,y,epochs=10):
        for i in range(epochs):
            ycap = self.forward(x)
            print(f"epoch: {i} | loss: {np.mean((y-ycap)**2)}")
            self.backward(y,ycap)
            
net = NN(784)
net.addlayer(100,"sig")
net.addlayer(10,"sig")