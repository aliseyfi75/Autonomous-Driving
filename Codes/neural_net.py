import numpy as np
import findMin
from sklearn.neural_network import MLPRegressor

# helper functions to transform between one big vector of weights
# and a list of layer parameters of the form (W,b) 
def flatten_weights(weights):
    return np.concatenate([w.flatten() for w in sum(weights,())])

def unflatten_weights(weights_flat, layer_sizes):
    weights = list()
    counter = 0
    for i in range(len(layer_sizes)-1):
        W_size = layer_sizes[i+1] * layer_sizes[i]
        b_size = layer_sizes[i+1]
        
        W = np.reshape(weights_flat[counter:counter+W_size], (layer_sizes[i+1], layer_sizes[i]))
        counter += W_size

        b = weights_flat[counter:counter+b_size][None]
        counter += b_size

        weights.append((W,b))  
    return weights

def log_sum_exp(Z):
    Z_max = np.max(Z,axis=1)
    return Z_max + np.log(np.sum(np.exp(Z - Z_max[:,None]), axis=1)) # per-colmumn max

class NeuralNet():
    # uses sigmoid nonlinearity
    def __init__(self, hidden_layer_sizes, classification = True, lammy=1, max_iter=100, verbose = True):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.lammy = lammy
        self.max_iter = max_iter
        self.classification = classification
        self.verbose = verbose

    def funObj(self, weights_flat, X, y):
        weights = unflatten_weights(weights_flat, self.layer_sizes)

        activations = [X]
        for W, b in weights:
            Z = X @ W.T + b
            X = 1/(1+np.exp(-Z))
            activations.append(X)

        yhat = Z
        
        if self.classification: # softmax- TODO: use logsumexp trick to avoid overflow
            tmp = np.sum(np.exp(yhat), axis=1)
            # f = -np.sum(yhat[y.astype(bool)] - np.log(tmp))
            f = -np.sum(yhat[y.astype(bool)] - log_sum_exp(yhat))
            grad = np.exp(yhat) / tmp[:,None] - y
        else:  # L2 loss
            f = 0.5*np.sum((yhat-y)**2)  
            grad = yhat-y # gradient for L2 loss

        grad_W = grad.T @ activations[-2]
        grad_b = np.sum(grad, axis=0)

        g = [(grad_W, grad_b)]

        for i in range(len(self.layer_sizes)-2,0,-1):
            W, b = weights[i]
            grad = grad @ W
            grad = grad * (activations[i] * (1-activations[i])) # gradient of logistic loss
            grad_W = grad.T @ activations[i-1]
            grad_b = np.sum(grad,axis=0)

            g = [(grad_W, grad_b)] + g # insert to start of list

        g = flatten_weights(g)
        
        # add L2 regularization
        f += 0.5 * self.lammy * np.sum(weights_flat**2)
        g += self.lammy * weights_flat 
        
        return f, g

    
    def fit(self, X, y):
        if y.ndim == 1:
            y = y[:,None]
            
        self.layer_sizes = [X.shape[1]] + self.hidden_layer_sizes + [y.shape[1]]
        # self.classification = y.shape[1]>1 # assume it's classification iff y has more than 1 column

        # random init
        scale = 0.01
        weights = list()
        for i in range(len(self.layer_sizes)-1):
            W = scale * np.random.randn(self.layer_sizes[i+1],self.layer_sizes[i])
            b = scale * np.random.randn(1,self.layer_sizes[i+1])
            weights.append((W,b))
        weights_flat = flatten_weights(weights)

        # utils.check_gradient(self, X, y, len(weights_flat), epsilon=1e-6)
        weights_flat_new, f = findMin.findMin(self.funObj, weights_flat, self.max_iter, X, y, verbose=self.verbose)

        self.weights = unflatten_weights(weights_flat_new, self.layer_sizes)


    def predict(self, X):
        for W, b in self.weights:
            Z = X @ W.T + b
            X = 1/(1+np.exp(-Z))
        if self.classification:
            return np.argmax(Z,axis=1)
        else:
            return Z

class NeuralNet_SGD(NeuralNet):
    # uses sigmoid nonlinearity
    def __init__(self, hidden_layer_sizes, lammy=1, batch_size=500, epochs=10, verbose=True):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.lammy = lammy
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose


    def fit(self, X, y):
        if y.ndim == 1:
            y = y[:, None]

        self.layer_sizes = [X.shape[1]] + self.hidden_layer_sizes + [y.shape[1]]
        self.classification = y.shape[1] > 1  # assume it's classification iff y has more than 1 column

        # random init
        scale = 0.01
        weights = list()
        for i in range(len(self.layer_sizes) - 1):
            W = scale * np.random.randn(self.layer_sizes[i + 1], self.layer_sizes[i])
            b = scale * np.random.randn(1, self.layer_sizes[i + 1])
            weights.append((W, b))
        weights_flat = flatten_weights(weights)

        # utils.check_gradient(self, X, y, len(weights_flat), epsilon=1e-6)
        weights_flat_new, f = findMin.findMin_SGD(self.funObj, weights_flat, self.epochs, X, y, self.batch_size, alpha=0.001, verbose=self.verbose)

        self.weights = unflatten_weights(weights_flat_new, self.layer_sizes)

class NeuralNet_Chain():
    def __init__(self, hidden_layer_sizes, classification = True, lammy=1, s=2, max_iter=100, verbose=True):
        self.hidden_layer_sizes = hidden_layer_sizes    # number of neurons for each model in chain
        self.lammy = lammy  # regularization hyper-param for each model in chain
        self.max_iter = max_iter    # maximum number of iterations
        self.classification = classification    # classification or regression indicator
        self.s = s  # chain stride
        self.verbose = verbose

    def fit(self, X, y):

        _, self.k = y.shape # output dimension of chained model

        # repeating the hyper-parameters when are not assigned
        while self.lammy.shape[0] != np.int(self.k/self.s):
            self.lammy = np.append(self.lammy,self.lammy[-1])

        # repeating the hyper-parameters when are not assigned
        while self.max_iter.shape[0] != np.int(self.k / self.s):
            self.max_iter = np.append(self.max_iter, self.max_iter[-1])

        # repeating the hyper-parameters when are not assigned
        while len(self.hidden_layer_sizes) != np.int(self.k/self.s):
            self.hidden_layer_sizes.append(self.hidden_layer_sizes[-1])

        # fitting dependent models repetitively
        models = []
        model = NeuralNet(hidden_layer_sizes=self.hidden_layer_sizes[0],classification=self.classification, lammy=self.lammy[0],max_iter=self.max_iter[0], verbose=self.verbose)
        model.fit(X,y[:,0:self.s])
        models.append(model)

        for i in range(1,np.int(self.k/self.s)):

            model = NeuralNet(hidden_layer_sizes=self.hidden_layer_sizes[i], classification=self.classification,
                              lammy=self.lammy[i], max_iter=self.max_iter[i], verbose=self.verbose)
            model.fit(np.concatenate((X,y[:,0:i*self.s]),axis=1), y[:, i*self.s:(i+1)*self.s])
            models.append(model)

        self.models = models

    def predict(self, X):

        # going forward through the chain
        y = self.models[0].predict(X)
        for i in range(1,np.int(self.k/self.s)):
            y_new = self.models[i].predict(np.concatenate((X,y),axis=1))
            y = np.concatenate((y,y_new),axis=1)

        return y

class NeuralNet_Chain_skLearn():
    def __init__(self, hidden_layer_sizes, classification = True, lammy=1, s=2, max_iter=100, verbose=True, activation='relu'):
        self.hidden_layer_sizes = hidden_layer_sizes    # number of neurons for each model in chain
        self.lammy = lammy  # regularization hyper-param for each model in chain
        self.max_iter = max_iter    # maximum number of iterations
        self.classification = classification    # classification or regression indicator
        self.s = s  # chain stride
        self.verbose = verbose
        self.activation = activation    # activation function

    def fit(self, X, y):

        _, self.k = y.shape # output dimension of chained model

        # repeating the hyper-parameters when are not assigned
        while self.lammy.shape[0] != np.int(self.k/self.s):
            self.lammy = np.append(self.lammy,self.lammy[-1])

        # repeating the hyper-parameters when are not assigned
        while self.max_iter.shape[0] != np.int(self.k / self.s):
            self.max_iter = np.append(self.max_iter, self.max_iter[-1])

        # repeating the hyper-parameters when are not assigned
        while len(self.hidden_layer_sizes) != np.int(self.k/self.s):
            self.hidden_layer_sizes.append(self.hidden_layer_sizes[-1])

        # fitting dependent models repetitively
        models = []
        model = MLPRegressor(hidden_layer_sizes=self.hidden_layer_sizes[0],activation=self.activation,alpha=self.lammy[0],max_iter=self.max_iter[0],verbose=self.verbose)
        model.fit(X,y[:,0:self.s])
        models.append(model)
        for i in range(1,np.int(self.k/self.s)):

            model = MLPRegressor(hidden_layer_sizes=self.hidden_layer_sizes[i], activation=self.activation,
                                 solver='sgd', batch_size=X.shape[0], alpha=self.lammy[i], max_iter=self.max_iter[i],
                                 verbose=self.verbose)

            model.fit(np.concatenate((X,y[:,0:i*self.s]),axis=1), y[:, i*self.s:(i+1)*self.s])
            models.append(model)

        self.models = models

    def predict(self, X):

        # going forward through the chain
        y = self.models[0].predict(X)
        for i in range(1,np.int(self.k/self.s)):
            y_new = self.models[i].predict(np.concatenate((X,y),axis=1))
            y = np.concatenate((y,y_new),axis=1)

        return y

class NeuralNet_Multiple():
    def __init__(self, hidden_layer_sizes, classification = True, lammy=1, s=2, max_iter=100, verbose=True):
        self.hidden_layer_sizes = hidden_layer_sizes    # number of neurons for each model
        self.lammy = lammy  # regularization hyper-param for each model
        self.max_iter = max_iter    # maximum number of iterations
        self.classification = classification    # classification or regression indicator
        self.s = s  # chain stride
        self.verbose = verbose

    def fit(self, X, y):

        _, self.k = y.shape # output dimension of multiple model

        # repeating the hyper-parameters when are not assigned
        while self.lammy.shape[0] != np.int(self.k/self.s):
            self.lammy = np.append(self.lammy,self.lammy[-1])

        # repeating the hyper-parameters when are not assigned
        while self.max_iter.shape[0] != np.int(self.k / self.s):
            self.max_iter = np.append(self.max_iter, self.max_iter[-1])

        # repeating the hyper-parameters when are not assigned
        while len(self.hidden_layer_sizes) != np.int(self.k/self.s):
            self.hidden_layer_sizes.append(self.hidden_layer_sizes[-1])

        # fitting independent models repetitively
        models = []
        model = NeuralNet(hidden_layer_sizes=self.hidden_layer_sizes[0],classification=self.classification, lammy=self.lammy[0],max_iter=self.max_iter[0], verbose=self.verbose)
        model.fit(X,y[:,0:self.s])
        models.append(model)

        for i in range(1,np.int(self.k/self.s)):

            model = NeuralNet(hidden_layer_sizes=self.hidden_layer_sizes[i], classification=self.classification,
                              lammy=self.lammy[i], max_iter=self.max_iter[i], verbose=self.verbose)
            model.fit(X, y[:, i*self.s:(i+1)*self.s])
            models.append(model)

        self.models = models

    def predict(self, X):

        # going forward through several independent models
        y = self.models[0].predict(X)
        for i in range(1,np.int(self.k/self.s)):
            y_new = self.models[i].predict(X)
            y = np.concatenate((y,y_new),axis=1)

        return y

class NeuralNet_Multiple_skLearn():
    def __init__(self, hidden_layer_sizes, classification = True, lammy=1, s=2, max_iter=100, verbose=True, activation='relu'):
        self.hidden_layer_sizes = hidden_layer_sizes    # number of neurons for each model
        self.lammy = lammy  # regularization hyper-param for each model
        self.max_iter = max_iter    # maximum number of iterations
        self.classification = classification    # classification or regression indicator
        self.s = s
        self.verbose = verbose
        self.activation = activation

    def fit(self, X, y):

        _, self.k = y.shape # output dimension of multiple model

        # repeating the hyper-parameters when are not assigned
        while self.lammy.shape[0] != np.int(self.k/self.s):
            self.lammy = np.append(self.lammy,self.lammy[-1])

        # repeating the hyper-parameters when are not assigned
        while self.max_iter.shape[0] != np.int(self.k / self.s):
            self.max_iter = np.append(self.max_iter, self.max_iter[-1])

        # repeating the hyper-parameters when are not assigned
        while len(self.hidden_layer_sizes) != np.int(self.k/self.s):
            self.hidden_layer_sizes.append(self.hidden_layer_sizes[-1])

        # fitting independent models repetitively
        models = []
        model = MLPRegressor(hidden_layer_sizes=self.hidden_layer_sizes[0],activation=self.activation,solver='sgd',batch_size=X.shape[0],alpha=self.lammy[0],max_iter=self.max_iter[0],verbose=self.verbose)
        model.fit(X,y[:,0:self.s])
        models.append(model)

        for i in range(1,np.int(self.k/self.s)):

            model = MLPRegressor(hidden_layer_sizes=self.hidden_layer_sizes[i], activation=self.activation,
                                 solver='sgd', batch_size=X.shape[0], alpha=self.lammy[0], max_iter=self.max_iter[0],
                                 verbose=self.verbose)
            model.fit(X, y[:, i*self.s:(i+1)*self.s])
            models.append(model)

        self.models = models

    def predict(self, X):

        # going forward through several independent models
        y = self.models[0].predict(X)
        for i in range(1,np.int(self.k/self.s)):
            y_new = self.models[i].predict(X)
            y = np.concatenate((y,y_new),axis=1)

        return y