import os
import numpy as np
import pandas as pd
from scipy.optimize import approx_fprime
from scipy import stats
import add_features
import preprocess


class chained():
    
    def __init__(self, model, *args):
        
        self.model = model
        self.args = args
        
    def fit(self,X,Y):
        
        n,d = X.shape
        n,k = Y.shape
        k = int(k/2)
        self.k = k 
        print('k=',k)
        x_models = []
        y_models = [] 
        for i in range(k):
            model = self.model(*self.args)
            model.fit(np.concatenate((X,Y[:,0:2*i]), axis=1),Y[:,2*i])
            x_models.append(model)
            model = self.model(*self.args)
            model.fit(np.concatenate((X,Y[:,0:2*i]), axis=1),Y[:,2*i+1])
            y_models.append(model)
            print(i)
            
        self.x_models = x_models
        self.y_models = y_models
        
    def predict(self,X):
        n,d = X.shape
        x_models = self.x_models
        y_models = self.y_models
        k = self.k
        
        Y = np.zeros((n,2*k))
        
        for i in range(k):
            Y[:,2*i] = x_models[i].predict(np.concatenate((X,Y[:,0:2*i]), axis=1))
            Y[:,2*i+1] = y_models[i].predict(np.concatenate((X,Y[:,0:2*i]), axis=1))
            print(i)

        return Y
        
    
    
def get_Xtrain(mypath):
    X_train=[]

    directory = os.path.join(mypath,'train','X')
    for root,dirs,files in os.walk(directory):
        for file in files:
           if file.endswith(".csv"):
               with open(os.path.join(mypath,'train','X',file), 'rb') as f:
                   data = pd.read_csv(f) 
                   data = add_features.replace_agent(data)
                   data = add_features.empty_fix(data)
                   data = preprocess.clean_data_X(data)
                   # data = add_features.dist2agent(data)
                   # data = add_features.poly2(data)
                   data = add_features.speed_direction(data)
                   data = add_features.acceleration(data)
                   data = add_features.turning(data)
                   data = np.array(data)
                   data = data.flatten()
                   X_train.append(data.tolist())
                   print(file)
                   
    X_train = np.array(X_train)
                
    X_train = np.vectorize(float)(X_train)
    return X_train
    

def get_Xtest(mypath):
    
    X_test=[]
    directory = os.path.join(mypath,'test','X')
    for root,dirs,files in os.walk(directory):
        for file in files:
           if file.endswith(".csv"):
               with open(os.path.join(mypath,'test','X',file), 'rb') as f:
                   data = pd.read_csv(f) 
                   data = add_features.replace_agent(data)
                   data = add_features.empty_fix(data)
                   data = preprocess.clean_data_X(data)
                   # data = add_features.dist2agent(data)
                   # data = add_features.poly2(data)
                   data = add_features.speed_direction(data)
                   data = add_features.acceleration(data)
                   data = add_features.turning(data)
                   data = np.array(data)
                   data = data.flatten()
                   X_test.append(data.tolist())
                   print(file)
                   
    
                
                
    X_test = np.array(X_test)
                
    X_test = np.vectorize(float)(X_test)
    return X_test


def get_Xval(mypath):
    
    X_val=[]
    directory = os.path.join(mypath,'val','X')
    for root,dirs,files in os.walk(directory):
        for file in files:
           if file.endswith(".csv"):
               with open(os.path.join(mypath,'val','X',file), 'rb') as f:
                   data = pd.read_csv(f) 
                   data = add_features.replace_agent(data)
                   data = add_features.empty_fix(data)
                   data = preprocess.clean_data_X(data)
                   # data = add_features.dist2agent(data)
                   # data = add_features.poly2(data)
                   data = add_features.speed_direction(data)
                   data = add_features.acceleration(data)
                   data = add_features.turning(data)
                   data = np.array(data)
                   data = data.flatten()
                   X_val.append(data.tolist())
                   print(file)
                   
    
                
                
    X_val = np.array(X_val)
                
    X_val = np.vectorize(float)(X_val)
    return X_val



def get_ytrain(mypath):
    y_train=[]

    directory = os.path.join(mypath,'train','y')
    for root,dirs,files in os.walk(directory):
        for file in files:
           if file.endswith(".csv"):
               with open(os.path.join(mypath,'train','y',file), 'rb') as f:
                   data = pd.read_csv(f) 
                   data = preprocess.clean_data_y(data)
                   data = np.array(data)
                   data = data.flatten()
                   y_train.append(data.tolist())
                   print(file)
                   
    y_train = np.array(y_train)
    return y_train


def get_yval(mypath):
    y_val=[]

    directory = os.path.join(mypath,'val','y')
    for root,dirs,files in os.walk(directory):
        for file in files:
           if file.endswith(".csv"):
               with open(os.path.join(mypath,'val','y',file), 'rb') as f:
                   data = pd.read_csv(f) 
                   data = preprocess.clean_data_y(data)
                   data = np.array(data)
                   data = data.flatten()
                   y_val.append(data.tolist())
                   print(file)
                   
    y_val = np.array(y_val)
    return y_val

def create_output(location):
    
    xs = [str(i) + "_x_" + str(j)  for i in range(20) for j in range(1,31)]
    ys = [str(i) + "_y_" + str(j)  for i in range(20) for j in range(1,31)]

    ids = []
    for i in range(len(xs)):
        ids.append(xs[i])
        ids.append(ys[i])

    data = {'Id': ids,
            'location': location}

    df = pd.DataFrame (data, columns = ['Id','location'])
    df.to_csv('ytest.csv', index=False)

def standardize_cols(X, mu=None, sigma=None):
    # Standardize each column with mean 0 and variance 1
    n_rows, n_cols = X.shape

    if mu is None:
        mu = np.mean(X, axis=0)

    if sigma is None:
        sigma = np.std(X, axis=0)
        sigma[sigma < 1e-8] = 1.

    return (X - mu) / sigma, mu, sigma

def check_gradient(model, X, y, dimensionality, verbose=True):
    # This checks that the gradient implementation is correct
    w = np.random.rand(dimensionality)
    f, g = model.funObj(w, X, y)

    # Check the gradient
    estimated_gradient = approx_fprime(w,
                                       lambda w: model.funObj(w,X,y)[0],
                                       epsilon=1e-6)

    implemented_gradient = model.funObj(w, X, y)[1]

    if np.max(np.abs(estimated_gradient - implemented_gradient) > 1e-3):
        raise Exception('User and numerical derivatives differ:\n%s\n%s' %
             (estimated_gradient[:5], implemented_gradient[:5]))
    else:
        if verbose:
            print('User and numerical derivatives agree.')

def mode(y):
    """Computes the element with the maximum count

    Parameters
    ----------
    y : an input numpy array

    Returns
    -------
    y_mode :
        Returns the element with the maximum count
    """
    if len(y)==0:
        return -1
    else:
        return stats.mode(y.flatten())[0][0]
    