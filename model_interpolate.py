import numpy
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from scipy.io import netcdf
 
class MyModel(object):

    def __init__(self, nsites):
        self.nsites = nsites
        self.nparms =self.nsites*2
        self.parm_names =  []
        self.pmin = numpy.zeros([self.nparms], numpy.float)-0.5
        self.pmax = numpy.zeros([self.nparms], numpy.float)
        self.pdef = numpy.zeros([self.nparms], numpy.float)
        for i in range(0,self.nparms):
            if (i % 2 == 0):
                self.pmax[i] = 103.5
                self.pdef[i] = numpy.random.uniform(-0.5,103.5,1)
            else:
                self.pmax[i] = 70.5
                self.pdef[i] = numpy.random.uniform(-0.5,70.5,1)
        self.pstd = numpy.zeros(self.nparms, numpy.float)
        myfile = netcdf.netcdf_file('ACMEv0_0.9x1.25_GPP_NorthAmerica_022317.nc','r')
        myvar = (myfile.variables['GPP'])
        area  = (myfile.variables['area'])
        lai   = (myfile.variables['TLAI'])
        self.gpp = myvar[:,:,:].copy()
        self.obs = self.gpp[0,:,:]
        self.obs_err = self.obs*0.0+100.0/(24*3600.0*365.0)
        for x in range(0,104):
            for y in range(0,71):
                #normalize by area
                #if (lai[0,y,x] < 0.05):
                #  self.obs[y,x] = 1e36
                self.obs_err[y,x] = self.obs_err[y,x]*area[50,50]/area[y,x]
        for i in range(0,self.nsites):
            isgood = False
            #come up with some default points that are "in bounds"
            while (not isgood):
                testsite = numpy.random.uniform(0,1,2)
                xtest = int(numpy.rint(testsite[0]*103.9-0.5))
                ytest = int(numpy.rint(testsite[1]*70.9-0.5))
                if (abs(self.obs[ytest,xtest]) < 0.01):
                    isgood = True
                    self.pdef[i*2] = xtest*1.0
                    self.pdef[i*2+1] = ytest*1.0
                    self.parm_names.append('Sitex')
                    self.parm_names.append('Sitey')
        self.ylabel = 'GPP (umol m-2 s-1)'
        self.issynthetic = False

    def run(self,parms):
        x = numpy.array([])
        y = numpy.array([])
        z = numpy.array([])
        npoints = 0
        for p in range(0,self.nsites):
            xtest = int(numpy.rint(parms[p*2]))
            ytest = int(numpy.rint(parms[p*2+1]))
            gpptest = self.gpp[0,ytest, xtest]
            if (gpptest >= 0.0 and gpptest < 0.01):
              x = numpy.append(x,xtest)
              y = numpy.append(y,ytest)
              z = numpy.append(z,gpptest)
            npoints = npoints + 1
        xi = numpy.linspace(0,103,104)
        yi = numpy.linspace(0,70,71)
        gppi_nr = griddata((x,y), z, (xi[None,:], yi[:,None]), method='nearest')
        gppi_cb =  griddata((x,y), z, (xi[None,:], yi[:,None]), method='cubic', fill_value=1e10)
        gppi = gppi_cb
        for x in range(0,104):
            for y in range(0,71):
                if (self.gpp[0,y,x] > 1):
                    gppi[y,x] = 1e36
                elif (self.gpp[0,y,x] < 1 and abs(gppi[y,x]) > 1e5):
                    gppi[y,x] = gppi_nr[y,x]
        self.output = gppi
     
