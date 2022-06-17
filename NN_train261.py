import numpy as np
import h5py
import random as rn
import tensorflow as tf
import os
import scipy.io as sio
from math import sqrt
import matplotlib.pyplot as plt
# plt.switch_backend('agg')
from sklearn.preprocessing import MinMaxScaler,StandardScaler,Normalizer
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from keras.models import Sequential 
from keras.layers import Dense,Dropout
from keras import optimizers
from tensorflow.compat.v1.keras import backend as K
from keras.callbacks import EarlyStopping
from scipy.io import loadmat
import time


# ==== fix random seed for reproducibility =====
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(7)
rn.seed(1245)
tf.random.set_seed(89)
session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1,
                              inter_op_parallelism_threads=1, 
                              allow_soft_placement=True,
                              device_count = {'CPU' : 1, 'GPU' : 0})
sess = tf.compat.v1.Session(graph= tf.compat.v1.get_default_graph(), config=session_conf)
K.set_session(sess)


start = time.time()
# ==== Define variables =====
Npar = 10
Ns = 262
Ntrain = Ns
Nsvd = 10

print('------- Nsvd ------ is ', Nsvd)

# ====  load labeld samples =====
xdata = np.loadtxt('Par_s'+str(Ns)+'.dat')

## SIF is 292896xNs matrix including the 2009-2014 (72 months) outputs for 4068 land points.
ydata = np.load('SIF_s'+str(Ns)+'.npy')


print('input and output size: ',xdata.shape,ydata.shape)
# print('min and max of ydata', np.min(ydata), np.max(ydata))

# exit()

scalerx = MinMaxScaler().fit(xdata)
xdata = scalerx.transform(xdata)

xtrain = xdata[0:Ntrain,:]
ytrain = ydata[0:Ntrain,:]


# ==== use SVD to reduce output dimension ====
u, s, vh = np.linalg.svd(ytrain, full_matrices=False)

# === define QoI ====
Vsvd = vh[0:Nsvd,:]
SVDytrain = np.matmul(Vsvd,ytrain.transpose())
SVDytrain = SVDytrain.transpose()  #size[Ntrain,Nsvd]
print('size of ytrain and SVDytrain: ',ytrain.shape,SVDytrain.shape)

#sio.savemat('Vsvd.mat', mdict={'Vsvd': Vsvd})

# ========== Define NN model ===============
Nepoch = 433

MSE_train = np.zeros(Nepoch)

#---- fit model
model = Sequential()
model.add(Dense(100, input_dim=Npar, activation='relu'))#, kernel_initializer='VarianceScaling'))
model.add(Dense(100, activation='relu'))#, kernel_initializer='VarianceScaling'))
model.add(Dense(Nsvd)) 
adam = optimizers.Adam(lr=0.01)
model.compile(optimizer=adam, loss='mse')
 
for i in range(Nepoch):
    model.fit(xtrain, SVDytrain, epochs=1, batch_size=10, verbose=0)

    NNQoI_train = model.predict(xtrain)
    NNytrain = np.matmul(NNQoI_train,Vsvd)
    MSE_train[i] = mean_squared_error(ytrain,NNytrain)
    print('Training data: mse and R2 of Epoch%d is %8.3e' % (i,MSE_train[i]))   


# #========= save NN model to JSON ==========
# serialize model to JSON
model_json = model.to_json()
with open("model_Ntrain"+str(Ns)+".json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model_Ntrain"+str(Ns)+".h5")
print("Saved model to disk..........")

plt.plot(MSE_train,'r')
plt.yscale('log')
plt.legend()
plt.show()
plt.close()

