import numpy as np


class Ridge:

    def init(self, lambda):
        self.lambda = lambda
    
    def fit(self,X,y):
        n, d = X.shape
        lambda = self.lambda
        a = np.array([[1]]*n)
        X_new=np.append(X, a, axis=1)
        self.w = solve(X_new.T@X_new + lambda*np.eye(d+1), X_new.T@y)

    def predict(self, X):
        n, d = X.shape
        a = np.array([[1]]*n)
        X_new=np.append(X, a, axis=1)
        return X_new@self.w
    
    
def chained_Ridge(X_train, y_train, X_test):
    model_x = Ridge()
    model_x.fit(X_train, y_train[:,0])
    x_hat_train = model_x.predict(X_train)
    x_hat_test = model_x.predict(X_test)

    model_y = Ridge()
    model_y.fit(X_train, y_train[:,1])
    y_hat_train = model_y.predict(X_train)
    y_hat_test = model_y.predict(X_test)

    y_pred_train = np.concatenate((x_hat_train.reshape(-1,1), y_hat_train.reshape(-1,1)),axis=1)
    y_pred_test = np.concatenate((x_hat_test.reshape(-1,1), y_hat_test.reshape(-1,1)),axis=1)

    for i in range(2,60):
        XN_train = np.concatenate((X_train,y_pred_train[:,i-2].reshape(-1,1)),axis =1)
        XN_test = np.concatenate((X_test,y_pred_test[:,i-2].reshape(-1,1)), axis = 1)

        model = Ridge()
        model.fit(XN_train, y_train[:,i])
        hat_train = model.predict(XN_train)
        hat_test = model.predict(XN_test)

        y_pred_train = np.concatenate((y_pred_train, hat_train.reshape(-1,1)),axis=1)
        y_pred_test = np.concatenate((y_pred_test, hat_test.reshape(-1,1)),axis=1)
        
    return y_pred_train, y_pred_test