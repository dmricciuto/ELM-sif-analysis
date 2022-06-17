import numpy as np
import h5py
from math import sqrt
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler,StandardScaler,Normalizer
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from keras import optimizers
from keras.models import model_from_json
import scipy.io as sio
from scipy.io import loadmat


# ==== Define variables =====
Ns = 261
Nsvd = 10

# ====  load data =====
SIF = np.load('GOSAT_sif_rf_model.npy').transpose()
print('SIF size',SIF.shape)
Par = np.loadtxt('../Sample275/MCsamples.dat')
print('Par size ', Par.shape)

randinx = np.loadtxt('Randinx_261.dat').astype(int)
SIF = SIF[randinx,:]
Par = Par[randinx,:]

scalerx = MinMaxScaler().fit(Par)
Par = scalerx.transform(Par)


# ==== use SVD to reduce output dimension ====
u, s, vh = np.linalg.svd(SIF, full_matrices=False)

# === define QoI ====
Vsvd = vh[0:Nsvd,:]
SVD_SIF = np.matmul(Vsvd,SIF.transpose())
SVD_SIF = SVD_SIF.transpose()  


# ====== Load saved NN model =========
json_file = open('model_Ntrain261.json', 'r')
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("model_Ntrain261.h5")


# ============ calculate MSE of train and test data ============
NNQoI = model.predict(Par)
NN_SIF = np.matmul(NNQoI,Vsvd)
MSE = mean_squared_error(SIF,NN_SIF)
print('Size of NN estimate: ',NN_SIF.shape)
print('Training data: mse is %8.2e' %MSE)   

Ntest = len(Par)
R2 = np.zeros(Ntest)
for i in range(Ntest):
	R2[i] = r2_score(SIF[i,:],NN_SIF[i,:])

print('max and min R2', np.max(R2), np.min(R2))

plt.plot(R2,'b.-')
plt.show()
