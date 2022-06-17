import numpy as np
from netCDF4 import Dataset
import h5py
from math import sqrt
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler,StandardScaler,Normalizer
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from keras import optimizers
from keras.models import model_from_json
from scipy.io import loadmat
import time

class MyModel(object):
    
    def __init__(self, var='FPSN'):
        if var == 'FPSN':
          self.name = 'GPP_surrogate'
          self.myvar = var
        elif var == 'SIF':
          self.name = 'SIF surrogate'
          self.myvar = var
        self.nparms = 10
        self.parm_names = ['flnr', 'mbbopt', 'bbbopt','roota_par','vcmaxha','jmaxha','vcmaxse', \
                           'dayl_scaling','dleaf','xl'] 
        self.pdef = [0.12, 9.0, 10000, 2.0, 70000, 60000, 670, 2.0, 0.04, 0.0]
        self.pmin = [0.03, 4.0,  1000, 1.0, 50000, 40000, 640, 0.0, 0.01,-0.6]
        self.pmax = [0.25,13.0, 40000,10.0, 90000, 80000, 700, 2.5, 0.10, 0.8]
        self.issynthetic = False

        # ==== Define variables =====
        if (var == 'FPSN'):
          self.Ntime = 168
        elif (var == 'SIF'):
          self.Ntime = 72  #84
        #Nloc = 5663
        Ns = 1000
        Ntrain = 262
        Nsvd = 10

        # ====  load traing data for scaling =====
        #train = np.loadtxt('mcsamples_20200221_261.txt')
        train = np.loadtxt('Par_s262.dat')
        for p in range(0,10):
            self.pmin[p] = min(train[:,p])
            self.pmax[p] = max(train[:,p])

        if (var == 'FPSN'):
          rh5py = h5py.File(self.myvar+'month275.hdf5', 'r')
          ydata = rh5py[self.myvar+'month275']
          ydata = ydata[:,:]  #to make dataset ydata as an array
          ytrain = np.delete(ydata,201,0)   #Remove a bad sample point
        elif (var == 'SIF'):
          #readmat = loadmat('./SIF_out4068_s261.mat')
          #ydata = readmat['SIF_out4068_s261']
          ydata = np.loadtxt('SIF_s262.dat') 
          ytrain = ydata[:,:]
          
        #print('max and min of ydata', np.max(ydata),np.min(ydata))

        # ==== use SVD to reduce output dimension ====
        u, s, vh = np.linalg.svd(ytrain, full_matrices=False)
        self.Vsvd = vh[0:Nsvd,:]

        #json_file = open('./trained_models/'+self.myvar+'_model_Ntrain261.json', 'r')
        json_file = open('./trained_models/SIF_GOSAT_brt_Ntrain262.json','r')
        model_json = json_file.read()
        json_file.close()
        self.svdmodel = model_from_json(model_json)
        #self.svdmodel.load_weights('./trained_models/'+self.myvar+"_model_Ntrain261.h5")
        self.svdmodel.load_weights('./trained_models/SIF_GOSAT_brt_Ntrain262.h5')

    def get_obs(self, xpoint=45, ypoint=45):
        #get the observation
        if xpoint < 72:
            xpointm = xpoint+72
        else:
            xpointm = xpoint-72
        self.obs = np.zeros([self.Ntime])
        if (self.myvar == 'FPSN'):
          for y in range(2000,2014):
            temp = Dataset('observations/GPP.RS_METEO.FP-ALL.MLM-ALL.METEO-CRUJRA_v1.144_96.monthly.'+ \
                  str(y)+'.nc','r')
            self.obs[(y-2000)*12:(y-1999)*12]=temp['GPP'][:,96-ypoint-1,xpointm]
            temp.close()
            #if (y == 2013):
            #    print(temp['lat'][96-ypoint-1], temp['lon'][xpointm], self.obs)
          self.obs_err = self.obs.copy()*0.0+0.5
        elif (self.myvar == 'SIF'):
          temp = Dataset('observations/GOME2_sif.nc','r')
          self.obs[:] = temp['observed_sif'][0:72,xpoint,ypoint]
          self.obs_err = self.obs.copy()*0.0+0.1
          temp.close()
        self.nobs = len(self.obs)
        self.x = np.cumsum(np.ones([self.nobs]))

    def get_indices(self, xpoint=45, ypoint=45):
        #get the observation
        if xpoint < 72:
            xpointm = xpoint+72
        else:
            xpointm = xpoint-72
        self.out_indices=[]
        self.xpoint = xpoint
        self.ypoint = ypoint
        #==== Get output indices for a gridcell of interest ====
        if (self.myvar == 'FPSN'):
          RawOut = Dataset('./model_output/'+self.myvar+'_full_ensemble_275.nc4','r')
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

          Nloc = 0
          for j in range(0,nlat):
            for i in range(0,nlon):
              if (Mask[j,i] == 1 and max(RawOut.variables[self.myvar][0,:,j,i].flatten()) > 0):
               iind.append(i)
               jind.append(j)
               Nloc = Nloc+1
          RawOut.close()
          ReadMask.close()
          #print(str(Nloc)+' valid land points')
          for t in range(0,self.Ntime):
            for i in range(0,Nloc):
              if (jind[i] == 96-ypoint-1 and iind[i]==xpointm):
                 self.out_indices.append(t*Nloc+i)
          #print(self.out_indices)
          self.landmask = 1
          if (max(self.obs[:]) <= 0.0 or len(self.out_indices) == 0):
            self.landmask = 0

        elif (self.myvar == 'SIF'):
          #RawOut = Dataset('./model_output/'+self.myvar+'_full_ensemble_rf_275.nc4','r')
          RawOut = Dataset('./model_output/SIF_full_ensemble_brt_GOSAT_275.nc4','r')
          #SIF=RawOut['modelled_sif'][:,96:168,:,:,0]
          nlon = RawOut.dimensions['lon'].size # nlon=144
          nlat = RawOut.dimensions['lat'].size # nlat=96
          Ncell = nlon*nlat
          #SIFout = np.reshape(SIF, (275, self.Ntime, nlon*nlat))
          #print('size of SIFout: ', np.shape(SIFout))
          #AveSIF = np.mean(SIFout,axis=0)
          #print('size of AveSIF: ', np.shape(AveSIF))
          RawOut.close()
          # ============ mapping back to the original domain (144,96)
          #ValidInx = np.loadtxt('./ValidInx.dat').astype(int)
          #Domain = np.zeros((self.Ntime,Ncell))
          XYInx = np.loadtxt('./XYindices.dat').astype(int)
          nland = max(XYInx.shape)
          print(XYInx.shape)
          stop
          # ==== extract nonNaN simulation samples
          # for 144*96 gridcells, there are 4068 nonNaN cells
          #common = np.arange((Ncell)) #save nonNaN grid cells index
          #print(ValidInx)
          #for t in range(self.Ntime):
          #  MatrixB = np.argwhere(~np.isnan(AveSIF[t,:])).flatten()
          #  common = list(set(common) & set(MatrixB))
          #for t in range(self.Ntime):
          #  inum = 0
          #  for i in ValidInx:
          #    if int(i/96) == xpoint and i % 96 == ypoint:
          #      #for t in range(self.Ntime):
          #      self.out_indices.append(t*len(ValidInx)+inum) 
          #      #self.out_indices.append(self.Ntime*len(common)+t)
          #    inum=inum+1
          for t in range(self.Ntime):
            for n in range(0,nland):
              if XYInx[n,0] == xpoint and XYInx[n,1] == ypoint:
                self.out_indices.append(t*nland+inum)

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
       #ValidInx = np.loadtxt('./ValidInx.dat').astype(int)
       #Domain = np.zeros((self.Ntime,144*96))
       #ReOneSample = np.reshape(NNytrain[0,:],[self.Ntime,4068])
       #Domain[:,ValidInx] = ReOneSample
       #GridDomain = np.reshape(Domain,(self.Ntime,144,96))
       #self.output = GridDomain[:,int(self.xpoint),int(self.ypoint)]
       end = time.time()
       #print('time', end - start)
