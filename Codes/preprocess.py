import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory


# from sklearn.naive_bayes import GaussianNB
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.ensemble import StackingClassifier
from statsmodels.tsa.ar_model import AutoReg
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor

# from random_forest import *
# from naive_bayes import *
# from stacking import *


import warnings
warnings.filterwarnings('ignore')

#%%

def clean_data_X(data):
    data = data.drop(columns = ['time step'])
    
    data = data.replace({" car": 1, " pedestrian/bicycle": 2})
    data = data.replace({" agent": 1, " others": 2})
    
    for i in range(10):
        data = data.drop(columns = [' id' +str(i), ' present'+str(i), ' role'+str(i)])

    
    return data

# clean_data_y needs to be implemented

def clean_data_y(data):
    size = 30    
    if size != data.shape[0]:
        diff=size-data.shape[0]    
        x = data[' x'].values
        y = data[' y'].values
        t = data['time step'].values

        model_x = AutoReg(x, 5)
        model_y = AutoReg(y, 5)
        model_t = AutoReg(t, 5)
        predictions_x = model_x.fit().predict(start=len(x), end=len(x)+diff-1, dynamic=False)
        predictions_y = model_y.fit().predict(start=len(y), end=len(y)+diff-1, dynamic=False)
        predictions_t = model_t.fit().predict(start=len(t), end=len(t)+diff-1, dynamic=False)
        d = np.concatenate((predictions_t.reshape(-1,1),predictions_x.reshape(-1,1),predictions_y.reshape(-1,1)),axis=1)
        d = pd.DataFrame(d, columns=['time step',' x', ' y'])
        data = data.append(d)
    
    return data.drop(columns = ['time step'])


def clean_y(y_test,y_val):
    
    n = y_test.shape[0]
    
    y = np.concatenate((y_test,y_val),axis=0)
    y1 = np.zeros(y.shape)
    index = np.array(range(0,60,2))
    xs = y[:,index]
    ys = y[:,index+1]
    model = MLPRegressor()
    model.fit(xs[:,:28],xs[:,28])
    x_new = model.predict(xs[:,1:29])
    model = MLPRegressor()
    model.fit(ys[:,:28],ys[:,28])
    y_new = model.predict(ys[:,1:29])
    
    y1[:,:58] = y[:,:58]
    y1[:,58] = x_new
    y1[:,59] = y_new
    return y1[:n], y1[n:]