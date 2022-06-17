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



# ==== Define variables =====
Npar = 10
Ns = 275
Ntrain = 200
Ntest = Ns-Ntrain 

ParName = ['flnr','mbbopt','bbbopt','roota_par','vcmaxha','jmaxha','vcmaxse','dayl_scaling','dleaf','xl']

# ====  load labeld samples =====
ParSample = np.loadtxt('Par_s275.dat')
minpar = np.min(ParSample, axis=0)
maxpar = np.max(ParSample, axis=0)
print('size of ParSample', ParSample.shape)


# ## SIF is 292896x275 matrix including the 2009-2014 (72 months) outputs for 4068 land points.
# SIF = np.loadtxt('SIF_s275.dat')

R2_train = np.loadtxt('R2_Train275.out')
R2_test = np.loadtxt('R2_Test275.out')

inx97 = np.argwhere(R2_test<0.97)
print('size of inx96',len(inx97))
# print(inx97)
print(Ntrain+inx97)

randinx = np.loadtxt('Randinx.dat')
randinx = np.delete(randinx, Ntrain+inx97)
np.savetxt('Randinx_261.dat',randinx, fmt='%d')
exit()

par_inx97 = np.delete(ParSample,(Ntrain+inx97),axis=0)
# SIF_inx97 = np.delete(SIF,(Ntrain+inx97),axis=0)
# print('par and SIF with R2_test>0.97 ', par_inx97.shape, SIF_inx97.shape)

tt=np.argwhere(par_inx97[:,0]<0.01)
print(tt)

exit()

np.savetxt('par_inx97.dat',par_inx97)
np.savetxt('SIF_inx97.dat',SIF_inx97)

fig = plt.figure(figsize=(20,4))
for i in range(5):
    plt.subplot(1,5,i+1)
    plt.plot(ParSample[:,i*2],ParSample[:,i*2+1],'r.')
    plt.plot(ParSample[Ntrain+inx97,i*2],ParSample[Ntrain+inx97,i*2+1],'b*')
    plt.xlim(minpar[i*2],maxpar[i*2])
    plt.ylim(minpar[i*2+1],maxpar[i*2+1])
    plt.xlabel(ParName[i*2],fontsize=12)
    plt.ylabel(ParName[i*2+1],fontsize=12)

fig.tight_layout() 
plt.savefig('ParSample.png', bbox_inches='tight', pad_inches=0.1, dpi=300)    
plt.show()








