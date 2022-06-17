import numpy as np
from keras.models import model_from_json
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from scipy.io import loadmat
from scipy.io import savemat
from cartopy import config
import cartopy.crs as ccrs
import cartopy.feature as cpf

Ntime =  72  #84
Ns = 1000
Ntrain = 262
Nsvd = 10
Nparms = 10
var_plot = 'FPSN'

# ====  load traing data for scaling =====
train = np.loadtxt('Par_s262.dat')

print(train.shape)
pmin = np.zeros([Nparms])
pmax = np.zeros([Nparms])

for p in range(0,Nparms):
  pmin[p] = min(train[:,p])
  pmax[p] = max(train[:,p])

ydata = np.load(var_plot+'_s262.npy')
ytrain = ydata[:,:]
Nland = int(max(ydata.shape)/Ntime)

#np.save('SIF_s262.npy', ydata)
print(Nland, ' land points')
print('max and min of ydata', np.max(ydata),np.min(ydata))

# ==== use SVD to reduce output dimension ====
u, s, vh = np.linalg.svd(ytrain, full_matrices=False)
Vsvd = vh[0:Nsvd,:]

if (var_plot == 'SIF'):
  json_file = open('./trained_models/SIF_GOSAT_brt_Ntrain262.json','r')
  model_json = json_file.read()
  json_file.close()
  svdmodel = model_from_json(model_json)
  svdmodel.load_weights('./trained_models/SIF_GOSAT_brt_Ntrain262.h5')
elif (var_plot == 'FPSN'):
  json_file = open('./trained_models/FPSN_model_Ntrain262.json','r')
  model_json = json_file.read()
  json_file.close()
  svdmodel = model_from_json(model_json)
  svdmodel.load_weights('./trained_models/FPSN_model_Ntrain262.h5','r')

parms=train[1,:]

#Get the XY indices for a pft of interest
if (var_plot == 'SIF'):
  XYind = np.loadtxt('XYindices.dat')
elif (var_plot == 'FPSN'):
 XYind = np.loadtxt('XYindices_FPSN.dat')
#myind=[]
#nland = max(XYind.shape)
#for n in range(0,nland):
#  if (XYind[n,2] == 7):
#    myind.append(n)


n_ens=100
xin=np.zeros([n_ens,10], np.float)
SIFout=np.zeros([n_ens,144,96])+1.
doprior=True
surfdata = Dataset('surfdata.nc','r')
pct_pft = surfdata['PCT_NAT_PFT']
area = surfdata['AREA']
landfrac = surfdata['LANDFRAC_PFT']
pct_natveg = surfdata['PCT_NATVEG']
gsmat = loadmat('Zhu_globalMonthlyGS.mat')

pfts=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
gppsum=np.zeros([n_ens])
for pft in pfts:
 print(pft)
 try:
   thischain = np.loadtxt('chains_fluxtower/chain_pft'+str(pft)+'.txt')
   haschain=True
 except:
   haschain=False
 psamples = thischain[0:40000:int(40000/n_ens),:]
 print('running ensemble')
 for i in range(0,n_ens):
   if (haschain):
     parms[0] = psamples[i,0]
     parms[1] = psamples[i,1]
     parms[6] = psamples[i,2]  #6
     #parms[3] = psamples[i,3]
   else:
     parms[0] = np.random.uniform(0.01, 0.25)
     parms[1] = np.random.uniform(2, 13)
     parms[6] = np.random.uniform(640,700)
   #Default values
   parms[2] = 10000.
   parms[3] = 2.0
   parms[4] = 70000.
   parms[5] = 60000
   parms[7] = 2.0
   parms[8] = 0.04 
   parms[9] = 0.0
   if (doprior or haschain == False):
     parms[0] = np.random.uniform(0.01, 0.25)
     parms[1] = np.random.uniform(2, 13)
     parms[6] = np.random.uniform(640,700)
   for p in range(0,Nparms): 
    xin[i,p] = (parms[p] - pmin[p]) / (pmax[p] - pmin[p])
 NNQoI = svdmodel.predict(xin) # NN model simulation
 NNyout = np.matmul(NNQoI,Vsvd)

 for n in range(0,max(XYind.shape)):
   npts = 0
   xind = int(XYind[n,0])
   yind = int(XYind[n,1])
   pftmax = int(XYind[n,2])
   xindr = int(XYind[n,0])+72
   if (int(XYind[n,0]) >= 72):
     xindr = int(XYind[n,0])-72
   yindr = 96-int(XYind[n,1])

   #print(xind,yind,pftmax,pct_pft[pft,yindr,xindr])
   if (pct_pft[pft,yind,xind] > 0.01):  #for SIF:  yindr, xindr; for FPSN:  yind,xind
       SIFouttemp=np.zeros([n_ens])
       for m in range(0,72):
         thismonth = m % 12
         if (gsmat['globalMonthlyGS'][int(yindr*3.75),int(xindr*5),thismonth] > -0.5): #for SIF:  xind,yind; for FPSN:  xindr,yindr
           SIFouttemp[:] = SIFouttemp[:] + pct_pft[pft,yind,xind] * NNyout[:,Nland*m+n]/100.0 #for SIF:  yindr,xindr; for FPSN:  yind,xind
           npts = npts+1
       if (npts > 0):
         SIFout[:,xindr,yind] = SIFout[:,xindr,yind] + SIFouttemp/npts  #for SIF:  xind,yindr; for FPSN:  xindr,yind
       #SIFout[i,xindr,yindr] = SIFout[i,xindr,yindr]/npts 
       #SIFout[t] = SIFout[t] + np.sum(NNyout[0,t*nland+np.array(myind)])/len(myind)
       #gppsum[i] = gppsum[i] + pct_pft[pft,int(XYind[n,1]),int(XYind[n,0])] * np.sum(NNyout[0,n:Nland*Ntime:Nland])/Ntime/100.0 \
       #             *pct_natveg[int(XYind[n,1]),int(XYind[n,0])]/100.0*area[int(XYind[n,1]),int(XYind[n,0])] * landfrac[int(XYind[n,1]),int(XYind[n,0])] * 12.0*86400/1e6 * 365.0 * 1e6 / 1e15
 #print(gppsum)

FPSN={}
FPSN['mean'] = np.mean(SIFout,axis=0).transpose()
FPSN['std'] = np.std(SIFout,axis=0).transpose()
#savemat('SIF_post.mat',FPSN)

ax = plt.axes(projection=ccrs.PlateCarree())

lons=(np.cumsum(np.ones([144],float))-1)*2.5-180.
lats=(np.cumsum(np.ones([96],float))-1)*1.9-90
plt.contourf(lons, lats, np.mean(SIFout,axis=0).transpose(), levels=[0.1,0.5,1,1.5,2,3,4,5,6,7,8,10], transform=ccrs.PlateCarree())
ax.coastlines()
#ax.add_feature(cpf.OCEAN, facecolor="w")
ax.add_feature(cpf.NaturalEarthFeature('physical', 'ocean', '10m', edgecolor='face', facecolor='w'))
ax.coastlines()
plt.colorbar()
plt.show()

ax = plt.axes(projection=ccrs.PlateCarree())
plt.contourf(lons, lats, np.std(SIFout,axis=0).transpose(), levels=[0.05,0.1,0.2,0.3,0.4,0.5,0.75,1,1.5,2,3,4], transform=ccrs.PlateCarree())
ax.coastlines()
plt.colorbar()
plt.show()

plt.contourf(np.mean(SIFout,axis=0).transpose())
plt.colorbar()
plt.show()
plt.contourf(np.std(SIFout,axis=0).transpose())
plt.colorbar()
plt.show()



