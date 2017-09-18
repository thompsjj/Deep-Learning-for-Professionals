import numpy as np

class NeuralNetwork:
    "Simple neural network"
    
    def __init__(self):  
        # Network architecture      
        self.input_layer_size  = 2
        self.hidden_layer_size = 3
        self.output_layer_size = 1
        
        # Weights 
        self.W1 = np.random.randn(self.input_layer_size, self.hidden_layer_size)
        self.W2 = np.random.randn(self.hidden_layer_size, self.output_layer_size)
        
    def cost_function(self, X, y):
        "Compute cost for given X,y, use weights already stored in class."
        self.y_hat = self.forward(X)
        J = 0.5*sum((y-self.y_hat)**2)
        return J
    
    def cost_function_prime(self, X, y):
        "Compute derivative with respect to W and W2 for a given X and y:"
        self.y_hat = self.forward(X)
        
        delta3 = np.multiply(-(y-self.y_hat), self.sigmoid_prime(self.z3))
        dJdW2 = np.dot(self.a2.T, delta3)
        
        delta2 = np.dot(delta3, self.W2.T)*self.sigmoid_prime(self.z2)
        dJdW1 = np.dot(X.T, delta2)  
        
        return dJdW1, dJdW2
    
    def forward(self, X):
        "Propagate inputs though network"
        self.z2 = np.dot(X, self.W1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.W2)
        yHat = self.sigmoid(self.z3) 
        return yHat

    @staticmethod
    def sigmoid(z):
        "Define sigmoid activation function to scalar, vector, or matrix"
        return 1/(1+np.exp(-z)) 

    def sigmoid_prime(self,z):
        "Gradient of sigmoid"
        return np.exp(-z)/((1+np.exp(-z))**2)
    
    #####################################################
    # Helper Functions
    def get_params(self):
        "Transform W1 and W2 unrolled into vector"
        params = np.concatenate((self.W1.ravel(), self.W2.ravel()))
        return params
    
    def set_params(self, params):
        "Set W1 and W2 using single paramater vector."
        W1_start = 0
        W1_end = self.hidden_layer_size * self.input_layer_size
        self.W1 = np.reshape(params[W1_start:W1_end], (self.input_layer_size , self.hidden_layer_size))
        W2_end = W1_end + self.hidden_layer_size*self.output_layer_size
        self.W2 = np.reshape(params[W1_end:W2_end], (self.hidden_layer_size, self.output_layer_size))
        
    def compute_gradients(self, X, y):
        dJdW1, dJdW2 = self.cost_function_prime(X, y)
        return np.concatenate((dJdW1.ravel(), dJdW2.ravel()))
        
from scipy import optimize

class Trainer:

    def __init__(self, N):
        "Make Local reference to network"
        self.N = N
        
    def callbackF(self, params):
        self.N.set_params(params)
        self.J.append(self.N.cost_function(self.X, self.y))   
        
    def costFunctionWrapper(self, params, X, y):
        self.N.set_params(params)
        cost = self.N.cost_function(X, y)
        grad = self.N.compute_gradients(X,y)
        
        return cost, grad
        
    def train(self, X, y, maxiter=50):

        self.X = X
        self.y = y
        self.J = [] # Empty list to store costs
        
        params0 = self.N.get_params() # Intial parameters at epoch 0

        options = {'maxiter': maxiter, 'disp' : True}
        _res = optimize.minimize(self.costFunctionWrapper,
                                 params0,
                                 jac=True,
                                 method='BFGS',
                                 args=(X, y), 
                                 options=options,
                                 callback=self.callbackF)
        self.N.set_params(_res.x)
        self.optimizationResults = _res
from scipy import optimize

class Trainer:

    def __init__(self, N):
        "Make Local reference to network"
        self.N = N
        
    def callbackF(self, params):
        self.N.set_params(params)
        self.J.append(self.N.cost_function(self.X, self.y))   
        
    def costFunctionWrapper(self, params, X, y):
        self.N.set_params(params)
        cost = self.N.cost_function(X, y)
        grad = self.N.compute_gradients(X,y)
        
        return cost, grad
        
    def train(self, X, y, maxiter=50):

        self.X = X
        self.y = y
        self.J = [] # Empty list to store costs
        
        params0 = self.N.get_params() # Intial parameters at epoch 0

        options = {'maxiter': maxiter, 'disp' : True}
        _res = optimize.minimize(self.costFunctionWrapper,
                                 params0,
                                 jac=True,
                                 method='BFGS',
                                 args=(X, y), 
                                 options=options,
                                 callback=self.callbackF)
        self.N.set_params(_res.x)
        self.optimizationResults = _res