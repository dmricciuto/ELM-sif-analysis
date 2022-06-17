import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset

#Grouped box plot
xlocations = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
width=0.4
#positions_group1 = [x-(width+0.01) for x in xlocations]
positions_group1 = xlocations
#positions_group3 = [x+(width+0.01) for x in xlocations]

allchains=np.zeros([3,60000,15])
pfts = [1,2,3,4,5,6,7,8,10,11,12,13,14,15]
for pft in pfts:
  thischain=np.loadtxt('chain_pft'+str(pft)+'.txt')
  print(pft, thischain.shape, thischain[1:10,6])
  allchains[0,:,pft-1] = thischain[:,0]
  allchains[1,:,pft-1] = thischain[:,1]
  allchains[2,:,pft-1] = thischain[:,6]

fig = plt.figure(1, figsize=(9, 6))
defparms = Dataset('clm_params_c180524.nc','r')
#flnr
ax = fig.add_subplot(311)
c1 ='black'   #GOSAT
ax.boxplot(allchains[0,:,:], showfliers=False, positions=positions_group1,widths=width, \
  patch_artist = True, boxprops=dict(color=c1,facecolor='none'), capprops=dict(color=c1), \
  whiskerprops=dict(color=c1), medianprops=dict(color=c1))
x=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
ax.plot(x,defparms['flnr'][1:16],'ko')
ax.set_ylim(0.01,0.28)
ax.set_ylabel('flnr')
ax.xaxis.set_ticklabels([])
#mbbopt
ax = fig.add_subplot(312)
ax.boxplot(allchains[1,:,:], showfliers=False, positions=positions_group1,widths=width, \
  patch_artist = True, boxprops=dict(color=c1,facecolor='none'), capprops=dict(color=c1), \
  whiskerprops=dict(color=c1), medianprops=dict(color=c1))
ax.plot(x,defparms['mbbopt'][1:16],'ko')
ax.set_ylim(2,15)
ax.set_ylabel('mbbopt')
ax.xaxis.set_ticklabels([])
#vcmaxse
ax= fig.add_subplot(313)
ax.boxplot(allchains[2,:,:], showfliers=False, positions=positions_group1,widths=width, \
  patch_artist = True, boxprops=dict(color=c1,facecolor='none'), capprops=dict(color=c1), \
  whiskerprops=dict(color=c1), medianprops=dict(color=c1))
ax.plot(x,np.zeros([15])+670.0,'ko')
ax.set_ylim(630,710)
ax.set_ylabel('vcmaxse')
plt.xticks([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15], ['ENFT','ENFB','DNF','EBFTr','EBFT','DBFTr','DBFT','DBFB','EBSh','DBShT','DBShB','C3GA','C3G','C4G','CROP'])
plt.show()
