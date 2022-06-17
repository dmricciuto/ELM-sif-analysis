import numpy as np
from netCDF4 import Dataset
import h5py
from math import sqrt
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler,StandardScaler,Normalizer
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from keras import optimizers
from keras.models import model_from_json
import time

class MyModel(object):
    
    def __init__(self, xpoint=45, ypoint=45):
        self.name = 'GPP_surrogate'
        self.nparms = 10
        self.parm_names = ['flnr', 'mbbopt', 'bbbopt','roota_par','vcmaxha','jmaxha','vcmaxse', \
                           'dayl_scaling','dleaf','xl'] 
        self.pdef = [0.12, 9.0, 10000, 2.0, 70000, 60000, 670, 2.0, 0.04, 0.0]
        self.pmin = [0.03, 4.0,  1000, 1.0, 50000, 40000, 640, 0.0, 0.01,-0.6]
        self.pmax = [0.25,13.0, 40000,10.0, 90000, 80000, 700, 2.5, 0.10, 0.8]
        self.issynthetic = False

        #get the observation
        self.obs = np.zeros([168])
        for y in range(2000,2014):
          temp = Dataset('observations/GPP.RS_METEO.FP-ALL.MLM-ALL.METEO-CRUJRA_v1.144_96.monthly.'+ \
                str(y)+'.nc','r')
          if (xpoint < 72):
            self.obs[(y-2000)*12:(y-1999)*12]=temp['GPP'][:,96-ypoint-1,xpoint+72]
          else:
            self.obs[(y-2000)*12:(y-1999)*12]=temp['GPP'][:,96-ypoint-1,xpoint-72]
          #if (y == 2000):
          #    print(temp['lat'][96-ypoint-1], temp['lon'][xpoint+72])

        self.obs_err = self.obs.copy()*0.0+0.5
        self.nobs = len(self.obs)
        self.x = np.cumsum(np.ones([self.nobs]))

        # ==== Define variables =====
        Ntime = 168
        Nloc = 5663
        Ns = 1000
        Ntrain = 100
        Nsvd = 10
        myvar='FPSN'

        # ====  load traing data for scaling =====
        train = np.loadtxt('mcsamples_20200221_200.txt')
        for p in range(0,10):
            self.pmin[p] = min(train[:,p])
            self.pmax[p] = max(train[:,p])

        xeva = np.loadtxt('RanSample_5000.dat')
        xeva = xeva[0:Ns,:]
        print('size of xeva',xeva.shape)

        scalerx = MinMaxScaler().fit(train)
        xeva = scalerx.transform(xeva)

        rh5py = h5py.File(myvar+'month200.hdf5', 'r')
        ydata = rh5py[myvar+'month200']
        ydata = ydata[:,:]  #to make dataset ydata as an array
        ytrain = ydata[0:Ntrain,:]
        print('max and min of ydata', np.max(ydata),np.min(ydata))

        # ==== use SVD to reduce output dimension ====
        u, s, vh = np.linalg.svd(ytrain, full_matrices=False)
        self.Vsvd = vh[0:Nsvd,:]

        json_file = open('./trained_models/'+myvar+'_model_Ntrain100.json', 'r')
        model_json = json_file.read()
        json_file.close()
        self.svdmodel = model_from_json(model_json)
        self.svdmodel.load_weights('./trained_models/'+myvar+"_model_Ntrain100.h5")

        #==== Get output indices for a gridcell of interest ====
        RawOut = Dataset('./model_output/'+myvar+'_full_ensemble.nc','r')
        nlon = RawOut.dimensions['lon'].size # nlon=144
        nlat = RawOut.dimensions['lat'].size # nlat=96
        lon = RawOut.variables['lon'][:]
        lat = RawOut.variables['lat'][:]
        ReadMask = Dataset('landmask.nc', 'r')
        Mask = ReadMask.variables['mask']
        iind=[]
        jind=[]
        if (max(self.obs[:]) <= 0.0):
          self.landmask = 0

        for j in range(0,nlat):
          for i in range(0,nlon):
            if Mask[j,i] == 1:
               iind.append(i)
               jind.append(j)

        self.out_indices=[]
        for t in range(0,Ntime):
          for i in range(0,Nloc):
            if (xpoint < 72 and jind[i] == 96-ypoint-1 and iind[i]==xpoint+72):
               self.out_indices.append(t*Nloc+i)
            elif (xpoint >= 72 and jind[i] == 96-ypoint-1 and iind[i]==xpoint-72):
               self.out_indices.append(t*Nloc+i)
        self.landmask = 1
        if (max(self.obs[:]) <= 0.0 or len(self.out_indices) == 0):
          self.landmask = 0



    def run(self, parms):
       start = time.time()
       xtrain=np.zeros([1,10], np.float)
       for p in range(0,self.nparms):
         xtrain[0,p] = (parms[p] - self.pmin[p]) / (self.pmax[p] - self.pmin[p])
       NNQoI_train = self.svdmodel.predict(xtrain) # NN model simulation
       NNytrain = np.matmul(NNQoI_train,self.Vsvd)
       self.output = NNytrain[0,self.out_indices]
       end = time.time()
       #print('time', end - start)
