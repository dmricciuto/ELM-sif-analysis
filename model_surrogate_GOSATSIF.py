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
    
    def __init__(self, pft=1):
        self.name = 'SIF surrogate'
        self.nparms = 10
        self.parm_names = ['flnr', 'mbbopt', 'bbbopt','roota_par','vcmaxha','jmaxha','vcmaxse', \
                           'dayl_scaling','dleaf','xl'] 
        self.pdef = [0.12, 9.0, 10000, 2.0, 70000, 60000, 670, 2.0, 0.04, 0.0]
        self.pmin = [0.01, 2.0,  1000, 1.0, 50000, 40000, 640, 0.0, 0.01,-0.6]
        self.pmax = [0.25,13.0, 40000,10.0, 90000, 80000, 700, 2.5, 0.10, 0.8]
        self.issynthetic = False

        self.Ntime = 72 
        Ns = 1000
        Ntrain = 262
        Nsvd = 10

        # ====  load traing data for scaling =====
        train = np.loadtxt('Par_s262.dat')
        for p in range(0,10):
            self.pmin[p] = min(train[:,p])
            self.pmax[p] = max(train[:,p])

        ydata = np.loadtxt('SIF_s262.dat') 
        ytrain = ydata[:,:]
          
        # ==== load the trained NN-SVD surrogate model ====
        u, s, vh = np.linalg.svd(ytrain, full_matrices=False)
        self.Vsvd = vh[0:Nsvd,:]

        json_file = open('./trained_models/SIF_GOSAT_brt_Ntrain262.json','r')
        model_json = json_file.read()
        json_file.close()
        self.svdmodel = model_from_json(model_json)
        self.svdmodel.load_weights('./trained_models/SIF_GOSAT_brt_Ntrain262.h5')

    def get_obs(self, pft=1):
        #Get the XY indices for a pft of interest
        XYind = np.loadtxt('XYindices.dat')
        #sif_obs = Dataset('observations/GOME2_sif.nc','r')
        sif_obs = Dataset('observations/Remap_GOSAT_sif.nc','r')
        self.obs = [] #np.zeros([self.Ntime])
        self.obs_err = [] #np.zeros([self.Ntime])
        self.nland = max(XYind.shape)
        self.npft=[] #np.zeros([self.Ntime], int)
        self.out_indices=[]
        gsmat = loadmat('Zhu_globalMonthlyGS.mat')
        print(gsmat['globalMonthlyGS'][:,:,:].shape)
        #plt.contourf(gsmat['globalMonthlyGS'][:,:,3])
        #plt.colorbar()
        #plt.show()
        #stop
        npft=0
        for n in range(0, self.nland):
          if (XYind[n,2] == pft and XYind[n,3] > 40.0 and XYind[n,0]):
            x = XYind[n,0] #-72
            y = XYind[n,1] #96 - XYind[n,1]
            self.obs.append(0)
            self.obs_err.append(0.0)
            self.npft.append(0)
            for t in range(3,60):   #Was 66
              thismonth=int(t % 12)
              if (sif_obs['SIF'][t-3,x,y] > 0.2 and gsmat['globalMonthlyGS'][int(y*3.75),int(x*5),thismonth] > 0.5):
                if (((pft == 2 or pft == 3 or pft == 8 or pft == 10 or pft == 11 or pft == 12) and  (thismonth >= 5 and thismonth <= 7)) or \
                    (pft == 1 or pft == 4 or pft == 5 or pft == 6 or pft == 7 or pft >= 13)):
                  #self.obs[t] = self.obs[t] + sif_obs['SIF'][t-3,x,y]
                  self.obs[npft] = self.obs[npft] + sif_obs['SIF'][t-3,x,y]
                  #Keep track of the number of good points for each month
                  #self.npft[t] = self.npft[t]+1
                  self.npft[npft] = self.npft[npft]+1
                  #Build the array of indices to sample from surrogate model
                  self.out_indices.append(t*self.nland+n)            
            if (self.npft[npft] > 6):
              self.obs[npft] = self.obs[npft]/self.npft[npft]
              #Calculate error term
              for t in range(3,60):
                thismonth=int(t % 12)
                if (sif_obs['SIF'][t-3,x,y] > 0.2 and gsmat['globalMonthlyGS'][int(y*3.75),int(x*5),thismonth] > 0.5):
                  if (((pft == 2 or pft == 3 or pft == 8 or pft == 10 or pft == 11 or pft == 12) and  (thismonth >= 5 and thismonth <= 7)) or \
                    (pft == 1 or pft == 4 or pft == 5 or pft == 6 or pft == 7 or pft >= 13)):
                    self.obs_err[npft] = self.obs_err[npft] + ((sif_obs['SIF'][t-3,x,y]-self.obs[npft])**2)/self.npft[npft]
              self.obs_err[npft] = np.sqrt(self.obs_err[npft]) #/self.npft[npft])
              npft=npft+1

            

        self.obs = np.array(self.obs[0:npft])
        self.obs_err = np.array(self.obs_err[0:npft])
        #Calculate error term
        #for n in range(0,self.nland):
        #  if (XYind[n,2] == pft):
        #    x = XYind[n,0] #-72
        #    y = XYind[n,1] #96 - XYind[n,1]
        #    for t in range(3,66):
        #      if (sif_obs['SIF'][t-3,x,y] > 0):
        #        self.obs_err[t] = self.obs_err[t] + ((sif_obs['SIF'][t-3,x,y]-self.obs[t])**2)/self.npft[t]
        #self.obs_err = np.sqrt(self.obs_err/self.npft)
        #for t in range(0,72):
        #  if (self.npft[t] < 2):
        #    self.obs[t] = np.NaN
        #    self.obs_err[t]=np.NaN
        #self.obs[0:4] = np.NaN
        #self.obs[60] = np.NaN
        #self.obs[65:] = np.NaN
        #if (pft == 1 or pft == 2 or pft == 3 or pft == 7 or pft >= 8):
        #  #NH non-growing season, April-Oct
        #  self.obs[0:72:12]=np.NaN  #Jan
        #  self.obs[1:72:12]=np.NaN  #Feb
        #  self.obs[2:72:12]=np.NaN  #Mar
        #  self.obs[10:72:12]=np.NaN #Nov
        #  self.obs[11:72:12]=np.NaN #Dec
        #if (pft == 2 or pft == 3 or pft == 8 or pft == 10 or pft == 11 or pft == 12):
        #  #Boreal/Arctic - Jun-Sep only
        #  self.obs[3:72:12]=np.NaN  #Apr
        #  self.obs[4:72:12]=np.NaN  #May
        #  #self.obs[8:72:12]=np.NaN  #Sep
        #  self.obs[9:72:12]=np.NaN  #Oct

        #print(self.npft)
        print('NPFT',npft)
        print(self.obs)
        print(self.npft)
        print(self.obs_err)
        #Build the array of corresponding indices to sample from surrogate model
        #self.out_indices=np.zeros([self.npft*self.Ntime],np.int)
        #for t in range(0,self.Ntime):
        #  self.out_indices[t*self.npft:(t+1)*self.npft]=t*self.nland+np.array(indices)

        sif_obs.close()
        self.nobs = len(self.obs)
        self.x = np.cumsum(np.ones([self.nobs]))

    def run(self, parms):
       xtrain=np.zeros([1,10], np.float)
       for p in range(0,self.nparms):
         xtrain[0,p] = (parms[p] - self.pmin[p]) / (self.pmax[p] - self.pmin[p])
       NNQoI_train = self.svdmodel.predict(xtrain) # NN model simulation
       NNytrain = np.matmul(NNQoI_train,self.Vsvd)
       #self.output = np.zeros([self.Ntime])
       self.output = np.zeros([len(self.obs)])
       output_all = NNytrain[0,self.out_indices]
       starti=0
       for n in range(0,len(self.obs)):
         self.output[n] = np.sum(output_all[starti:starti+self.npft[n]])/self.npft[n]
         starti=starti+self.npft[n]
