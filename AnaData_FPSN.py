# np.count_nonzero(~np.isnan(RawObs[t,:,:])) count the # non-NaN grid cells.
# np.argwhere(~np.isnan(LandValue)) find index of non-Nan elements
# list(set(A)|set(B)) find union of 1d array A and B
# list(set(A)&set(B))  find intersection of 1d array A and B
# convert 3d array to 2d: for A[l, m, n], np.reshape(A, (l*m, n)) 

# consider extra 75 samples draw from flnr[0,0.04] and mmbox[0.2,0.4]
# however, for very small flnr samples, we have small GPP and for these samples, the surrogate accuracy is very low
# So we consider flnr samples with values >0.01, which gives us 262 samples
# but the 215th sample has low R2 score, we remove the 215th sample and build surrogate on 261 samples.


import numpy as np
import random as rn
import numpy.ma as ma
import h5py
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.io import loadmat
from netCDF4 import Dataset


#--- simulation data has 275 samples where 13 samples has flnr<0.01
#Ns = 275 
start_month = 108
n_months = 72

#===== Read parameter samples, last 75 samples are newly add
#OriSamples = np.loadtxt('../Sample275/MCsamples.dat')
OriSamples = np.loadtxt('mcsamples_20200221_275.txt')
good = np.argwhere(OriSamples[:,0] >= 0.01)
OriSamples=OriSamples[good.squeeze(),:]
Ns = len(good[:])

# SIF is 292896x275 matrix including the 2009-2014 (72 months) outputs for 4068 land points.
# where 17466 data points having all the same 275 sample values

#Orginal ensemble file
#SIF_ensemble_file = Dataset('model_output/SIF_full_ensemble_brt_GOSAT_275.nc4','r')
#SIF_ensemble_data = SIF_ensemble_file['modelled_sif'][good.squeeze(),start_month:start_month+n_months,:,:,:]
FPSN_ensemble_file = Dataset('./model_output/FPSN_full_ensemble_275.nc4','r')
FPSN_ensemble_data=FPSN_ensemble_file['FPSN'][good.squeeze(),start_month:start_month+n_months,:,:]

xland=[]
yland=[]
nland=0

surfdata=Dataset('surfdata.nc','r')
pftfrac = surfdata['PCT_NAT_PFT']

#Figure out which points are land, write indices and dominant PFT
output=open('XYindices_FPSN.dat','w')
ctpft=np.zeros([17])
for x in range(0,144):
  for y in range(0,96):
    if (max(FPSN_ensemble_data[0,:,y,x]) > 0.01):
      xland.append(x) 
      yland.append(y)
      thispftfrac=pftfrac[:,y,x]
      thispft=0
      for i in range(1,17):
        if (thispftfrac[i] > 40 and thispftfrac[i] == max(thispftfrac)):
          thispft=i
          ctpft[i]=ctpft[i]+1
      output.write(str(xland[nland])+' '+str(yland[nland])+' '+str(thispft)+'\n')
      nland=nland+1
output.close()

#Convert to vector format
FPSN_ensemble_landonly = np.zeros([Ns,nland*n_months])
for n in range(0,nland):
  for t in range(0,n_months):
    FPSN_ensemble_landonly[:,t*nland+n] = FPSN_ensemble_data[:,t,yland[n],xland[n]]


#OriSIF = np.load('GOSAT_sif_rf_model.npy').transpose()
OriFPSN = FPSN_ensemble_landonly
print('OriFPSN size',OriFPSN.shape)

rn.seed(1245)

randinx = rn.sample(range(0, Ns), Ns)
print(randinx)
np.savetxt('Randinx.dat',randinx, fmt='%d')
#exit()

ParSamples = OriSamples[randinx,:]
FPSN = OriFPSN[randinx,:]
print('ParSamples and FPSN size ', ParSamples.shape, FPSN.shape)

np.savetxt('Par_s'+str(Ns)+'.dat',ParSamples)
np.save('FPSN_s'+str(Ns)+'.npy',FPSN)




