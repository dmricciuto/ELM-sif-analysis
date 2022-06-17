from netCDF4 import Dataset
import numpy as np


surfdata = Dataset('surfdata_2000.nc','r')
parms = Dataset('best_parms_GPP.nc','r')

landfrac = surfdata.variables['LANDFRAC_PFT'][:,:]
pct_natveg = surfdata.variables['PCT_NATVEG'][:,:]
pft = surfdata.variables['PCT_NAT_PFT'][:,:,:]
area = surfdata.variables['AREA'][:,:]
lon = surfdata.variables['LONGXY'][:,:]
lat = surfdata.variables['LATIXY'][:,:]
best_parms = parms.variables['best_parms'][:,:,:]

pftcells = np.zeros([10,17])
area_sum = np.zeros([17])
pftct = np.zeros([17])
allvals = np.zeros([10,17,5000])

output=open('pftdata.txt','w')
for p in range(1,17):
  for x in range(0,144):
    for y in range(0,96):
      if landfrac[y,x] > 0.9 and pct_natveg[y,x] > 90:
        if  ((pft[p,y,x] == np.max(pft[p,y,x])) and pft[p,y,x] > 40):
          output.write(str(p)+','+str(x)+','+str(y)+','+str(lon[y,x])+','+str(lat[y,x])+'\n')
          pftct[p] = pftct[p]+1
print(pftct)
output.close()
stop

for x in range(0,144):
  for y in range(0,96):
    if landfrac[y,x] > 0.8 and pct_natveg[y,x] > 40:
      for k in range(0,17):
        if pft[k,y,x] > 20:
          xp = x + 72
          if (x >= 72):
            xp = x-72
          if (max(best_parms[:,y,xp]) > 0 and lon[y,x] > 180):
            for p in range(0,10):
              pftcells[p,k] = pftcells[p,k]+best_parms[p,y,xp]*area[y,x]*landfrac[y,x]*pft[k,y,x]/100.
              allvals[p,k,int(pftct[k])] = best_parms[p,y,xp]
            pftct[k] = pftct[k]+1
            area_sum[k] = area_sum[k]+area[y,x]*landfrac[y,x]*pft[k,y,x]/100.

#for k in range(1,17):
#  for p in range(0,1):
#    print k,p, pftcells[p,k]/area_sum[k], np.std(allvals[p,k,0:int(pftct[k])])
#print pftct[1:17]
