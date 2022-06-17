import matplotlib.pyplot as plt
import numpy as np
import pickle
import model_surrogate as models
#from mpl_toolkits.basemap import Basemap
from netCDF4 import Dataset

colors=['b','g','k','r','c']
sif_all_outputs = np.zeros([16,300,96,144],np.float)
sif_prediction = np.zeros([300,96,144],np.float)
gpp_prediction = np.zeros([300,96,144],np.float)

default=Dataset(' 20200221_f19_f19_ICBCLM45BC.clm2.h0.2000-2014_default.nc','r')
optGPP=Dataset(' 20200221_f19_f19_ICBCLM45BC.clm2.h0.2000-2014_optGPP.nc','r')
optSIF=Dataset(' 20200221_f19_f19_ICBCLM45BC.clm2.h0.2000-2014_optSIF.nc','r')

surfdat = Dataset('./surfdata_2000.nc','r')
lat = np.zeros([96,144],np.float)
lon = np.zeros([96,144],np.float)
lat = surfdat['LATIXY'][:,:]
lon = surfdat['LONGXY'][:,:]
#lat[:,0:72] = surfdat['LATIXY'][:,72:144]
#lat[:,72:144] = surfdat['LATIXY'][:,0:72]
#lon[:,0:72] = surfdat['LONGXY'][:,72:144]
#lon[:,72:144] = surfdat['LONGXY'][:,0:72]


pct_pft_shift = surfdat['PCT_NAT_PFT'][:,:,:]
pct_pft = pct_pft_shift.copy()
pct_pft[:,:,0:72] = pct_pft_shift[:,:,72:144]
pct_pft[:,:,72:144] = pct_pft_shift[:,:,0:72]

do_predictions=False

mask = np.ones([96,144],np.float)
mask_orig = np.ones([96,144],np.float)
for j in range(0,96):
  for i in range(0,144):
      if (i < 72):
          ii=i+72
      else:
          ii=i-72
      if surfdat['LANDFRAC_PFT'][j,i] == 0:
          mask[j,ii] = np.NaN
      mask_orig[j,i] = np.NaN

if (do_predictions):
  #initialize a gpp model
  #model_fpsn = models.MyModel(var='FPSN')
  #get indices for an example point
  #model_fpsn.get_obs(pft=1)
  #model_fpsn.get_indices(xpoint = [45], ypoint=[45])

  myvars = ['SIF']
  mytype = ['post']   #prior, post
  chains={}
  #models={}
  for v in myvars:
   for p in range(1,16):
    print('Loading '+v+' info for pft '+str(p))
    if (p != 9):
      if ('SIF' in v):   #GOSAT only
        chains['pft'+str(p)] = np.loadtxt('output/chain_pft'+str(p)+'_GOSAT_SIF.txt')
      else:
        chains['pft'+str(p)] = np.loadtxt('output/chain_pft'+str(p)+'_'+v+'3p.txt')
        #models[v+'pft'+str(p)] = pickle.load(open('output/model_pft'+str(p)+'_'+v+'3p.pkl','rb'))
   for t in mytype:
    for p in range(1,16):
      fpsnmodel = models.MyModel(var='FPSN')
      fpsnmodel.get_obs(pft=p)
      fpsnmodel.get_indices(xpoint = fpsnmodel.pftxind, ypoint=fpsnmodel.pftyind)
      if (v == 'SIF'):
        sifmodel = models.MyModel(var='SIF')
        sifmodel.get_obs(pft=p)
        sifmodel.get_indices(xpoint = sifmodel.pftxind, ypoint=sifmodel.pftyind)
      for i in range(0,100):
       if (p != 9):
        pchain = chains['pft'+str(p)][i*200,:]
        if (t == 'prior'):
         pchain[0] = np.random.uniform(0.01,0.25)
         pchain[1] = np.random.uniform(2,13)
         pchain[6] = np.random.uniform(640,700)
        if (v == 'SIF'):
         sifmodel.run(pchain,globalSIF=True)
         sif_all_outputs[p,i,:,:] = sifmodel.output
         sif_prediction[i,:,:] = sif_prediction[i,:,:]+pct_pft[p,:,:]*sifmodel.output/100.0
        fpsnmodel.run(pchain,globalFPSN=True)
        gpp_prediction[i,:,0:72] = gpp_prediction[i,:,0:72]+pct_pft[p,:,0:72]*fpsnmodel.output[:,72:144]/100.0
        gpp_prediction[i,:,72:144] = gpp_prediction[i,:,72:144]+pct_pft[p,:,72:144]*fpsnmodel.output[:,0:72]/100.0
        print(v,t,p,i)
    np.save('FPSN_GOSAT_prediction_'+t+v+'.npy', gpp_prediction)
    if (v == 'SIF'):
      np.save('SIF_GOSAT_prediction_'+t+v+'.npy', sif_prediction)
  stop
else:
  gpp_prediction = np.load('GOME/FPSN_prediction_postSIF.npy')
  sif_prediction = np.load('GOME/SIF_prediction_postSIF.npy')
  allchains={}
  myvars = ['FPSN','SIF','SIF_GOSAT']
  for v in myvars:
    for p in range(1,16):
      if (p != 9):
        if ('GOSAT' in v):
          thischain = np.loadtxt('output/chain_pft'+str(p)+'_GOSAT_SIF.txt')
       
        else:
          thischain = np.loadtxt('output/chain_pft'+str(p)+'_'+v+'3p.txt')
        if (p == 1):
            allchains[v] = np.zeros([3,len(thischain[:,0]),15],np.float)
        allchains[v][0,:,p-1]=thischain[:,0]
        allchains[v][1,:,p-1]=thischain[:,1]
        allchains[v][2,:,p-1]=thischain[:,6]

#Grouped box plot
xlocations = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
width=0.2
positions_group1 = [x-(width+0.01) for x in xlocations]
positions_group2 = xlocations
positions_group3 = [x+(width+0.01) for x in xlocations]

fig = plt.figure(1, figsize=(9, 6))
defparms = Dataset('clm_params_c180524.nc','r')
#flnr
ax = fig.add_subplot(311)
c1 ='black'   #FPSN
c2 = 'red'    #GOME SIF
c3 = 'blue'   #GOSAT SIF
ax.boxplot(allchains['FPSN'][0,:,:], showfliers=False, positions=positions_group1,widths=width, \
  patch_artist = True, boxprops=dict(color=c1,facecolor='none'), capprops=dict(color=c1), \
  whiskerprops=dict(color=c1), medianprops=dict(color=c1))
ax.boxplot(allchains['SIF'][0,:,:], showfliers=False, positions=positions_group2,widths=width, \
  patch_artist = True, boxprops=dict(color=c2,facecolor='none'), capprops=dict(color=c2), \
  whiskerprops=dict(color=c2), medianprops=dict(color=c2))
ax.boxplot(allchains['SIF_GOSAT'][0,:,:], showfliers=False, positions=positions_group3,widths=width, \
  patch_artist = True, boxprops=dict(color=c3,facecolor='none'), capprops=dict(color=c3), \
  whiskerprops=dict(color=c3), medianprops=dict(color=c3))
x=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
ax.plot(x,defparms['flnr'][1:16],'ko')
ax.set_ylim(0.01,0.28)
ax.set_ylabel('flnr')
ax.xaxis.set_ticklabels([])
#mbbopt
ax = fig.add_subplot(312)
ax.boxplot(allchains['FPSN'][1,:,:], showfliers=False, positions=positions_group1,widths=width, \
  patch_artist = True, boxprops=dict(color=c1,facecolor='none'), capprops=dict(color=c1), \
  whiskerprops=dict(color=c1), medianprops=dict(color=c1))
ax.boxplot(allchains['SIF'][1,:,:], showfliers=False, positions=positions_group2,widths=width, \
  patch_artist = True, boxprops=dict(color=c2,facecolor='none'), capprops=dict(color=c2), \
  whiskerprops=dict(color=c2), medianprops=dict(color=c2))
ax.boxplot(allchains['SIF_GOSAT'][1,:,:], showfliers=False, positions=positions_group3,widths=width, \
  patch_artist = True, boxprops=dict(color=c3,facecolor='none'), capprops=dict(color=c3), \
  whiskerprops=dict(color=c3), medianprops=dict(color=c3))
ax.plot(x,defparms['mbbopt'][1:16],'ko')
ax.set_ylim(2,15)
ax.set_ylabel('mbbopt')
ax.xaxis.set_ticklabels([])
#vcmaxse
ax= fig.add_subplot(313)
ax.boxplot(allchains['FPSN'][2,:,:], showfliers=False, positions=positions_group1,widths=width, \
  patch_artist = True, boxprops=dict(color=c1,facecolor='none'), capprops=dict(color=c1), \
  whiskerprops=dict(color=c1), medianprops=dict(color=c1))
ax.boxplot(allchains['SIF'][2,:,:], showfliers=False, positions=positions_group2,widths=width, \
  patch_artist = True, boxprops=dict(color=c2,facecolor='none'), capprops=dict(color=c2), \
  whiskerprops=dict(color=c2), medianprops=dict(color=c2))
ax.boxplot(allchains['SIF_GOSAT'][2,:,:], showfliers=False, positions=positions_group3,widths=width, \
  patch_artist = True, boxprops=dict(color=c3,facecolor='none'), capprops=dict(color=c3), \
  whiskerprops=dict(color=c3), medianprops=dict(color=c3))
ax.plot(x,np.zeros([15])+670.,'ko')
ax.set_ylim(630,710)
ax.set_ylabel('vcmaxse')
plt.xticks([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15], ['ENFT','ENFB','DNF','EBFTr','EBFT','DBFTr','DBFT','DBFB','EBSh','DBShT','DBShB','C3GA','C3G','C4G','CROP'])
plt.show()

#--------------------------------End parameter plot---------------------------------------

gpp_sum=np.zeros([100],np.float)
gppfac = 12/1e6*3600*24
for n in range(0,100):
    gpp_sum[n]=np.sum(default['landfrac'][:,72:144]*surfdat['AREA'][:,72:144]*gpp_prediction[n,:,0:72]*gppfac*365*1e6/1e15)
    gpp_sum[n]=gpp_sum[n]+np.sum(default['landfrac'][:,0:72]*surfdat['AREA'][:,0:72]*gpp_prediction[n,:,72:144]*gppfac*365*1e6/1e15)

print(np.mean(gpp_sum),np.std(gpp_sum))
default_sum=np.ma.sum(np.ma.mean(default['FPSN'][:,:,:]*gppfac*default['landfrac'][:,:]*surfdat['AREA'][:,:]*365,axis=0))*1e6/1e15
optGPP_sum=np.ma.sum(np.ma.mean(optGPP['FPSN'][:,:,:]*gppfac*default['landfrac'][:,:]*surfdat['AREA'][:,:]*365,axis=0))*1e6/1e15
optSIF_sum=np.ma.sum(np.ma.mean(optSIF['FPSN'][:,:,:]*gppfac*default['landfrac'][:,:]*surfdat['AREA'][:,:]*365,axis=0))*1e6/1e15

print(default_sum, optGPP_sum, optSIF_sum)
#for n in range(0,100):
#  plt.plot(np.nanmean(gpp_prediction[n,:,:]*gppfac*mask*365,axis=1),linewidth=0.5,color='gray')
#  #plt.plot(np.nanmean
#print(default['FPSN'][6,:,:])
#plt.plot(np.ma.mean(np.ma.mean(default['FPSN'][:,:,:]*gppfac*365,axis=0),axis=1),color='red')
#plt.plot(np.ma.mean(np.ma.mean(optGPP['FPSN'][:,:,:]*gppfac*365,axis=0),axis=1),color='blue')
#plt.plot(np.ma.mean(np.ma.mean(optSIF['FPSN'][:,:,:]*gppfac*365,axis=0),axis=1),color='black')
plt.contourf(lon, lat, np.ma.mean(optSIF['FPSN'][:,:,:]*gppfac*365,axis=0), levels=[0,100,200,300,400,500,750,1000,1250,1500,2000,2500,3000,3500,4000])
plt.colorbar()  
plt.show()



#-----------------------------End latitude plot ----------------------------------



#print(gpp_prediction[0,20,:])
#m = Basemap(projection='robin',lon_0=0,resolution='c')
#m.drawcoastlines()
#m.fillcontinents(color='coral',lake_color='aqua')
# draw parallels and meridians.
#m.drawparallels(np.arange(-90.,120.,30.))
#m.drawmeridians(np.arange(0.,360.,60.))
#m.drawmapboundary(fill_color='aqua')
#x, y = m(lon*180./np.pi, lat*180./np.pi)
#print(lon)
plt.contourf(lon, lat, np.std(gpp_prediction[0:100,:,:]*gppfac*mask*365,axis=0), levels=[0,10,20,30,40,50,75,100,150,200,300,400])
plt.colorbar()
plt.show()
plt.contourf(lon,lat,np.mean(gpp_prediction[0:100,:,:]*gppfac*mask*365,axis=0), levels=[0,100,200,300,400,500,750,1000,1250,1500,2000,2500,3000,3500,4000])
plt.colorbar()
plt.show()


plt.contourf(lon, lat, np.std(sif_prediction[0:100,:,:]*mask,axis=0))
plt.colorbar()
plt.show()
plt.contourf(lon,lat,np.mean(sif_prediction[0:100,:,:]*mask,axis=0))
plt.colorbar()
plt.show()#  output2 = model.output
#  plt.contourf(output2-output1)
#  plt.colorbar()
# plt.show()


#  print(np.shape(chain)[0]
flnr, edges = np.histogram(chain[:,1],flnrbin) 
flnr = flnr / np.shape(chain)[0]
plt.plot((edges[1:]+edges[0:-1])/2,flnr,color=colors[p-4])
plt.axvline(x=model.parms_best[1],color=colors[p-4],linestyle='--')
plt.show()
