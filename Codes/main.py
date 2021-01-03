import numpy as np
import pandas as pd
from neural_net import NeuralNet_Chain, NeuralNet_Multiple
import pickle
import utils

# Load the preprocessed data
X_train = pd.read_csv('../data/X_train3.csv')
X_train = np.array(X_train.iloc[:,1:])

y_train = pd.read_csv('../data/y_train3.csv')
y_train = np.array(y_train.iloc[:,1:])

X_val = pd.read_csv('../data/X_val3.csv')
X_val = np.array(X_val.iloc[:,1:])

y_val = pd.read_csv('../data/y_val3.csv')
y_val = np.array(y_val.iloc[:,1:])

X_test = pd.read_csv('../data/X_test3.csv')
X_test = np.array(X_test.iloc[:,1:])

# Data dimension
print("X_train  \n n = %d, d = %d" %(X_train.shape[0],X_train.shape[1]))
print("y_train  \n n = %d, d = %d" %(y_train.shape[0],y_train.shape[1]))
print("X_val  \n n = %d, d = %d" %(X_val.shape[0],X_val.shape[1]))
print("y_val  \n n = %d, d = %d" %(y_val.shape[0],y_val.shape[1]))
print("X_test  \n n = %d, d = %d" %(X_test.shape[0],X_test.shape[1]))

# Normalize the data
X_train_normalized, mu_x, sigma_x = utils.standardize_cols(X_train)
y_train_normalized, mu_y, sigma_y = utils.standardize_cols(y_train)
X_val_normalized, _, _ = utils.standardize_cols(X_val,mu_x,sigma_x)
y_val_normalized, _, _ = utils.standardize_cols(y_val,mu_y,sigma_y)
X_test_normalized, _, _ = utils.standardize_cols(X_test,mu_x,sigma_x)

# 100 Neurons, s=2, lammy=1, max_iter=1000, normalized data, chained dependent models

model = NeuralNet_Chain(hidden_layer_sizes=[[100]],classification=False, lammy=np.array([1]),s=2, max_iter=np.array([1000]), verbose=True)

model.fit(X_train_normalized,y_train_normalized)

# Compute training error
y_pred = model.predict(X_train_normalized)
train_error = np.mean(np.sqrt(np.sum((y_pred - y_train_normalized)**2,axis=1)))
print("train error: %f" %train_error)

# Compute validation error
y_pred = model.predict(X_val_normalized)
validation_error = np.mean(np.sqrt(np.sum((y_pred - y_val_normalized)**2,axis=1)))
print("validation error: %f" %validation_error)

# Save the model to disk
filename = 'chained_NN_ONE_Layer_100N_s2_Normalized.sav'
pickle.dump(model, open(filename, 'wb'))

# 100 Neurons, s=1, lammy=1, max_iter=1000, normalized data, chained dependent models

model = NeuralNet_Chain(hidden_layer_sizes=[[100]],classification=False, lammy=np.array([1]),s=1, max_iter=np.array([1000]), verbose=True)

model.fit(X_train_normalized,y_train_normalized)

# Compute training error
y_pred = model.predict(X_train_normalized)
train_error = np.mean(np.sqrt(np.sum((y_pred - y_train_normalized)**2,axis=1)))
print("train error: %f" %train_error)

# Compute validation error
y_pred = model.predict(X_val_normalized)
validation_error = np.mean(np.sqrt(np.sum((y_pred - y_val_normalized)**2,axis=1)))
print("validation error: %f" %validation_error)

# Save the model to disk
filename = 'chained_NN_ONE_Layer_100N_s1_Normalized.sav'
pickle.dump(model, open(filename, 'wb'))

# 100 Neurons, s=60, lammy=1, max_iter=1000, normalized data, chained dependent models

model = NeuralNet_Chain(hidden_layer_sizes=[[100]],classification=False, lammy=np.array([1]),s=60, max_iter=np.array([1000]), verbose=True)

model.fit(X_train_normalized,y_train_normalized)

# Compute training error
y_pred = model.predict(X_train_normalized)
train_error = np.mean(np.sqrt(np.sum((y_pred - y_train_normalized)**2,axis=1)))
print("train error: %f" %train_error)

# Compute validation error
y_pred = model.predict(X_val_normalized)
validation_error = np.mean(np.sqrt(np.sum((y_pred - y_val_normalized)**2,axis=1)))
print("validation error: %f" %validation_error)

# Save the model to disk
filename = 'chained_NN_ONE_Layer_100N_s60_Normalized.sav'
pickle.dump(model, open(filename, 'wb'))


# 100 Neurons, s=2, lammy=1, max_iter=1000, not normalized data, chained dependent models

model = NeuralNet_Chain(hidden_layer_sizes=[[100]],classification=False, lammy=np.array([1]),s=2, max_iter=np.array([1000]), verbose=True)

model.fit(X_train,y_train)

# Compute training error
y_pred = model.predict(X_train)
train_error = np.mean(np.sqrt(np.sum((y_pred - y_train_normalized)**2,axis=1)))
print("train error: %f" %train_error)

# Compute validation error
y_pred = model.predict(X_val)
validation_error = np.mean(np.sqrt(np.sum((y_pred - y_val)**2,axis=1)))
print("validation error: %f" %validation_error)

# Save the model to disk
filename = 'chained_NN_ONE_Layer_100N_s2_NotNormalized.sav'
pickle.dump(model, open(filename, 'wb'))

# 100 Neurons, s=1, lammy=1, max_iter=1000, not normalized data, chained dependent models

model = NeuralNet_Chain(hidden_layer_sizes=[[100]],classification=False, lammy=np.array([1]),s=1, max_iter=np.array([1000]), verbose=True)

model.fit(X_train,y_train)

# Compute training error
y_pred = model.predict(X_train)
train_error = np.mean(np.sqrt(np.sum((y_pred - y_train)**2,axis=1)))
print("train error: %f" %train_error)

# Compute validation error
y_pred = model.predict(X_val)
validation_error = np.mean(np.sqrt(np.sum((y_pred - y_val)**2,axis=1)))
print("validation error: %f" %validation_error)

# Save the model to disk
filename = 'chained_NN_ONE_Layer_100N_s1_NotNormalized.sav'
pickle.dump(model, open(filename, 'wb'))

# 100 Neurons, s=60, lammy=1, max_iter=1000, not normalized data, chained dependent models

model = NeuralNet_Chain(hidden_layer_sizes=[[100]],classification=False, lammy=np.array([1]),s=60, max_iter=np.array([1000]), verbose=True)

model.fit(X_train,y_train)

# Compute training error
y_pred = model.predict(X_train)
train_error = np.mean(np.sqrt(np.sum((y_pred - y_train)**2,axis=1)))
print("train error: %f" %train_error)

# Compute validation error
y_pred = model.predict(X_val)
validation_error = np.mean(np.sqrt(np.sum((y_pred - y_val)**2,axis=1)))
print("validation error: %f" %validation_error)

# Save the model to disk
filename = 'chained_NN_ONE_Layer_100N_s60_NotNormalized.sav'
pickle.dump(model, open(filename, 'wb'))

# 500 Neurons, s=60, lammy=1, max_iter=1000, not normalized data, chained dependent models

model = NeuralNet_Chain(hidden_layer_sizes=[[100]],classification=False, lammy=np.array([1]),s=60, max_iter=np.array([1000]), verbose=True)

model.fit(X_train,y_train)

# Compute training error
y_pred = model.predict(X_train)
train_error = np.mean(np.sqrt(np.sum((y_pred - y_train)**2,axis=1)))
print("train error: %f" %train_error)

# Compute validation error
y_pred = model.predict(X_val)
validation_error = np.mean(np.sqrt(np.sum((y_pred - y_val)**2,axis=1)))
print("validation error: %f" %validation_error)

# Save the model to disk
filename = 'chained_NN_ONE_Layer_500N_s60_NotNormalized.sav'
pickle.dump(model, open(filename, 'wb'))


# 100 Neurons, s=2, lammy=1, max_iter=1000, not normalized data, multiple independent models

model = NeuralNet_Multiple(hidden_layer_sizes=[[100]],classification=False, lammy=np.array([1]),s=2, max_iter=np.array([1000]), verbose=True)

model.fit(X_train,y_train)

# Compute training error
y_pred = model.predict(X_train)
train_error = np.mean(np.sqrt(np.sum((y_pred - y_train_normalized)**2,axis=1)))
print("train error: %f" %train_error)

# Compute validation error
y_pred = model.predict(X_val)
validation_error = np.mean(np.sqrt(np.sum((y_pred - y_val)**2,axis=1)))
print("validation error: %f" %validation_error)

# Save the model to disk
filename = 'independent_NN_ONE_Layer_100N_s2_NotNormalized.sav'
pickle.dump(model, open(filename, 'wb'))

# 100 Neurons, s=1, lammy=1, max_iter=1000, not normalized data, multiple independent models

model = NeuralNet_Multiple(hidden_layer_sizes=[[100]],classification=False, lammy=np.array([1]),s=1, max_iter=np.array([1000]), verbose=True)

model.fit(X_train,y_train)

# Compute training error
y_pred = model.predict(X_train)
train_error = np.mean(np.sqrt(np.sum((y_pred - y_train)**2,axis=1)))
print("train error: %f" %train_error)

# Compute validation error
y_pred = model.predict(X_val)
validation_error = np.mean(np.sqrt(np.sum((y_pred - y_val)**2,axis=1)))
print("validation error: %f" %validation_error)

# Save the model to disk
filename = 'Independent_NN_ONE_Layer_100N_s1_NotNormalized.sav'
pickle.dump(model, open(filename, 'wb'))

# Load the best model based on validation error
filename = 'chained_NN_ONE_Layer_500N_s60_NotNormalized.sav'
model = pickle.load(open(filename, 'rb'))

# fit on all train and validation data
model.fit(np.concatenate((X_train,X_val),axis=0),np.concatenate((y_train,y_val),axis=0))

# Prediction
y_pred = model.predict(X_test)

# Convert the prediction to Kaggle submition format
xs = [str(i) + "_x_" + str(j) for i in range(20) for j in range(1, 31)]
ys = [str(i) + "_y_" + str(j) for i in range(20) for j in range(1, 31)]

ids = []
for i in range(len(xs)):
    ids.append(xs[i])
    ids.append(ys[i])

data = {'Id': ids,
        'location': y_pred.reshape(-1,1).flatten()}

df = pd.DataFrame(data, columns=['Id', 'location'])
df.to_csv('ytest.csv', index=False)

print("finished!")
