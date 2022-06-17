import numpy
from scipy.stats import norm
from scipy.optimize import differential_evolution
from scipy.optimize import minimize
import model_DALEC as models
import os, math, random
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

def posterior(parms):
    line = 0
    #Uniform priors
    prior = 1.0
    for j in range(0,model.nparms):
        if (parms[j] < model.pmin[j] or parms[j] > model.pmax[j]):
            prior = 0.0
    #for j in range(0,model.nsites):
    #    xtest = int(numpy.rint(parms[j*2]))
    #    ytest = int(numpy.rint(parms[j*2+1]))
    #    if (xtest < 0 or xtest > 103 or ytest < 0 or ytest > 70):
    #        prior = 0.0
    #    elif (abs(model.obs[ytest,xtest]) > 1e5):
    #        prior = 0.0

    #Gaussian priors
    #prior = 0
    #for j in range(0,model.nparms):
    #    prior = prior + norm.logpdf(parms[j], loc=model.pdef[j], scale=model.pstd[j])

    post = prior
    if (prior > 0.0):
        model.run(parms)
        myoutput = model.output.flatten()
        myobs    = model.obs.flatten()
        myerr    = model.obs_err.flatten()
        for v in range(0,len(myoutput)):
            if (abs(myobs[v]) < 1e5):
                resid = (myoutput[v] - myobs[v])
                ri = (resid/myerr[v])**2
                li = -1.0 * numpy.log(2.0*numpy.pi)/2.0 - \
                     numpy.log(myerr[v]) - ri/2.0
                post = post + li
    else:
        post = -9999999
    return(post)

def posterior_min(parms):
    post = posterior(parms)
    #print post
    return(-1.0*post)


#Specific cases for comparison with Dan Lu's paper
def posterior_Cauchy(parms):
    post = 0
    for i in range(0,model.nparms):
        li = -numpy.log(1+parms[i]**2)
        post = post+li
    return post

def posterior_multivariate(parms):
    post = 0
    myloc = numpy.zeros(50,numpy.float)
    mycov = numpy.zeros([50,50],numpy.float)
    for i in range(0,50):
        for j in range(0,50):
            if (i == j):
                mycov[i,j] = 0.1*(i+1)*(i+1)
            else:
                mycov[i,j] = 0.05*(i+1)*(j+1)
    post = sum(sum(norm.logpdf(parms,loc=myloc,scale=mycov)))
    return post

#---------------------------------- QPSO --------------------------------------------------

def QPSO(npop, niter):

    beta_l = 0.6
    beta_u = 0.6
        
    nevals = 0
    x = numpy.zeros([npop,model.nparms],numpy.float)    #parameters for each pop
    fx = numpy.zeros(npop,numpy.float)                #costfunc for each pop
    xbestall = numpy.zeros(model.nparms,numpy.float)
    fxbestall = 9999999
    
    #randomize starting positions, get posteriors
    for n in range(0,npop):
        for p in range(0,model.nparms):
            x[n,p] = numpy.random.uniform(model.pmin[p],model.pmax[p],1)
        fx[n] = -1.0*posterior(x[n,:])
        nevals = nevals+1
    xbest = numpy.copy(x)            #Best parms for each pop so far
    fxbest = numpy.copy(fx)          #Best costfunc for each pop so far
    xbestall[:] = numpy.copy(x[0,:])    #Overall best parms so far
    fxbestall = numpy.copy(fx[0])    #OVerall best costfunc so far

    for n in range(1,npop):
        if (fxbest[n] < fxbestall):
            #get overall best parameters and function values
            xbestall[:] = numpy.copy(x[n,:])
            fxbestall = numpy.copy(fx[n])

    mbest = numpy.zeros(model.nparms)
    for i  in range(1,niter):
        beta = beta_u - (beta_u-beta_l)*i*1.0/niter
        xlast = numpy.copy(x)
        for p in range(0, model.nparms):
            mbest[p] = numpy.sum(xbest[:,p])/npop
        for n in range(0,npop):
            isvalid = False
            while (not isvalid):
                u = numpy.random.uniform(0,1,1)
                v = numpy.random.uniform(0,1,1)
                w = numpy.random.uniform(0,1,1)
                pupdate = u * xbest[n,:] + (1.0-u)*xbestall[:]
                betapro = beta * numpy.absolute(mbest[:] - xlast[n,:])
                x[n,:] = pupdate + (-1.0 ** numpy.ceil(0.5+v))*betapro*(-numpy.log(w))
                isvalid = True
                for p in range(0,model.nparms):
                    if (x[n,p] < model.pmin[p] or x[n,p] > model.pmax[p]):
                        isvalid=False

            fx[n] = -1.0*posterior(x[n,:])
            nevals = nevals+1
        for n in range(0,npop):
            if (fx[n] < fxbest[n]):
                xbest[n,:] = numpy.copy(x[n,:])
                fxbest[n] = numpy.copy(fx[n])
            if (fx[n] < fxbestall):
                xbestall[:] = numpy.copy(x[n,:])
                fxbestall = numpy.copy(fx[n])
                print i, n, fxbestall
    print 'QPSO best: ', fxbestall
    print xbestall
    return xbestall




#-------------------------------- MCMC ------------------------------------------------------

def MCMC(parms, nevals, type='uniform', nburn=1000, burnsteps=10):
    
    post_best = -99999
    post_last = -99999
    accepted_step = 0
    accepted_tot  = 0
    nparms     = model.nparms
    #parms      = numpy.zeros(nparms)
    parm_step  = numpy.zeros(nparms)
    chain      = numpy.zeros((nparms+1,nevals))
    chain_prop = numpy.zeros((nparms,nevals))
    chain_burn = numpy.zeros((nparms,nevals))
    #output     = numpy.zeros((model.nobs,nevals))
    mycov      = numpy.zeros((nparms,nparms))

    for p in range(0,nparms):
        #Starting step size = 1% of prior range
        #parm_step[p] = 2.4**2/nparms * (model.pmax[p]-model.pmin[p])
        parm_step[p] = 0.01 * (model.pmax[p]-model.pmin[p])
        #parms[p] = numpy.random.uniform(parms[p]-parm_step[p],parms[p]+parm_step[p],1)
        parms[p] = model.pdef[p]
        #parms_sens = numpy.copy(parms)
        #vary this parameter by one step
        #parms_sens[p] = parms_sens[p]+parm_step[p]
        #post_sens = posterior(parms_sens)
        #use 1D sensitivities to decrease the step sizes accordingly
        #print p, numpy.absolute(post_def - post_sens)
        #if (numpy.absolute(post_def - post_sens) > 1.0):
        #    parm_step[p] = parm_step[p]/(numpy.absolute(post_def - post_sens))
    for i in range(0,nparms):
        mycov[i,i] = parm_step[i]**2

    parm_last = parms
    scalefac = 1.0

    for i in range(0,nevals):

         #update proposal step size
        if (i > 0 and (i % nburn) == 0 and i < burnsteps*nburn):
            acc_ratio = float(accepted_step) / nburn
            mycov_step = numpy.cov(chain_prop[0:nparms,accepted_tot- \
                                              accepted_step:accepted_tot])
            mycov_chain = numpy.cov(chain_burn[0:nparms,(accepted_tot/4):accepted_tot])
            thisscalefac = 1.0
            if (acc_ratio <= 0.2):
                thisscalefac = max(acc_ratio/0.3, 0.15)
            elif (acc_ratio > 0.4):
                thisscalefac = min(acc_ratio/0.3, 2.5)
            scalefac = scalefac * thisscalefac
            for j in range(0,nparms):
                for k in range(0,nparms):
                    if (acc_ratio > 0.05):
                        mycov[j,k] = mycov_chain[j,k] * scalefac
                            #if (j == k):
                            #mycov[j,k] =
                                #scalefac* max(mycov_chain[j,j] / \
                                       #  mycov_step[j,j], 1) * mycov_step[j,j]
                    else:
                        #if (j == k):
                        mycov[j,k] = thisscalefac * mycov[j,k]
                    if (j == k):
                        print j, scalefac,mycov[j,j]/(parm_step[j]**2)


            print 'BURNSTEP', i/nburn, acc_ratio, thisscalefac, scalefac
            mycov_step = numpy.cov(chain_prop[0:nparms,accepted_tot- \
                                                  accepted_step:accepted_tot])
            #print(numpy.corrcoef(chain[0:4,i-nburn:i]))
            accepted_step = 0
    
    
        if (i == burnsteps*nburn):
        #Parameter chain plots
            for p in range(0,nparms):
                fig = plt.figure()
                xchain = numpy.cumsum(numpy.ones(nburn*burnsteps))
                plt.plot(xchain, chain[p,0:nburn*burnsteps])
                plt.xlabel('Evaluations')
                plt.ylabel(model.parm_names[p])
                if not os.path.exists('./plots/chains'):
                    os.makedirs('./plots/chains')
                plt.savefig('plots/chains/burnin_chain_'+model.parm_names[p]+'.pdf')
                plt.close(fig)
    
    
    
        #get proposal step
        parms = numpy.random.multivariate_normal(parm_last, mycov)
   
        #------- run the model and calculate log likelihood -------------------
        post = posterior(parms)
        
        #determine whether proposal step is accepted
        if ( (post - post_last < numpy.log(random.uniform(0,1)))):
            #if not accepted, go back to previous step
            for j in range(0,nparms):
                parms[j] = parm_last[j]
        else:
            #proposal step is accepted
            post_last = post
            accepted_tot = accepted_tot+1
            accepted_step = accepted_step+1
            chain_prop[0:nparms,accepted_tot] = parms-parm_last
            chain_burn[0:nparms,accepted_tot] = parms
            parm_last = parms
            #keep track of best solution so far
            if (post > post_best):
                post_best = post
                parms_best = parms
                print post_best
                output_best = model.output

        #populate the chain matrix
        for j in range(0,nparms):
            chain[j][i] = parms[j]
        chain[nparms][i] = post_last
        #for j in range(0,model.nobs):
        #    output[j,i] = model.output[j]

        if (i % 1000 == 0):
            print ' -- '+str(i)+' --\n'

    return parms_best
    print "Computing statistics"
    chain_afterburn = chain[0:nparms,nburn*burnsteps:]
    chain_sorted = chain_afterburn
    #output_sorted = output[0:model.nobs,nburn*burnsteps:]
    #output_sorted.sort()

    numpy.savetxt('chain.txt', numpy.transpose(chain_afterburn))
    #Print out some statistics
    print "Best parameters"
    print parms_best
    print "Posterior for best solution: ", post_best
    print "Parameter correlation matix"
    parmcorr =  numpy.corrcoef(chain_afterburn)
    print parmcorr
    print ''
    #parameter correlation plots (threshold correlations)
    corr_thresh = 0.8
    for p1 in range(0,nparms-1):
      for p2 in range(p1+1,nparms):
        if (abs(parmcorr[p1,p2]) > corr_thresh):
          fig = plt.figure()
          plt.hexbin(chain_afterburn[p1,:],chain_afterburn[p2,:])
          cbar = plt.colorbar()
          cbar.set_label('bin count')
          plt.xlabel(model.parm_names[p1])
          plt.ylabel(model.parm_names[p2])

          plt.suptitle('r = '+str(parmcorr[p1,p2]))
          if not os.path.exists('./plots/corr'):
              os.makedirs('./plots/corr')
          plt.savefig('./plots/corr/corr_'+model.parm_names[p1]+'_'+model.parm_names[p2]+'.pdf')
          plt.close(fig)
    #Parameter chain plots
    for p in range(0,nparms):
        fig = plt.figure()
        xchain = numpy.cumsum(numpy.ones(nevals-nburn*burnsteps))
        plt.plot(xchain, chain_afterburn[p,:])
        plt.xlabel('Evaluations')
        plt.ylabel(model.parm_names[p])
        if not os.path.exists('./plots/chains'):
            os.makedirs('./plots/chains')
        plt.savefig('./plots/chains/chain_'+model.parm_names[p]+'.pdf')
        plt.close(fig)

    chain_sorted.sort()
    print "95% parameter confidence interval"
    for p in range(0,nparms):
        print model.parm_names[p], \
        chain_sorted[p,int(0.025*(nevals-nburn*burnsteps))], \
        chain_sorted[p,int(0.975*(nevals-nburn*burnsteps))]
    print "Ratio of accepted steps to total steps:"
    print float(accepted_tot)/nevals
    print ''
    print "95% prediction confidence interval"
    for p in range(0,model.nobs):
        print output_sorted[p,int(0.025*(nevals-nburn*burnsteps))], \
        output_sorted[p,int(0.975*(nevals-nburn*burnsteps))]
    #make parameter histogram plots
    for p in range(0,nparms):
        fig = plt.figure()
        n, bins, patches = plt.hist(chain_afterburn[p,:],25,normed=1)
        if (model.issynthetic):
            #plot actual parameters for synthetic data case
            plt.axvline(x=model.actual_parms[p],linewidth=2,color='r')
        plt.xlabel(model.parm_names[p])
        plt.ylabel('Probability Density')
        if not os.path.exists('./plots/pdfs'):
            os.makedirs('./plots/pdfs')
        plt.savefig('./plots/pdfs/'+model.parm_names[p]+'.pdf')
        plt.close(fig)

    #make prediction plots
    fig = plt.figure()
    ax=fig.add_subplot(111)
    ax.errorbar(model.x,model.obs, yerr=model.obs_err, label='Observations')
    ax.plot(model.x,output_best,'r', label = 'Model best')
    ax.plot(model.x,output_sorted[:,int(0.025*(nevals-nburn*burnsteps))], \
                 'k--', label='Model 95% CI')
    ax.plot(model.x,output_sorted[:,int(0.975*(nevals-nburn*burnsteps))],'k--')
    plt.xlabel(model.xlabel)
    plt.ylabel(model.ylabel)
    box = ax.get_position()
    ax.set_position([box.x0,box.y0,box.width*0.8,box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1,0.5), fontsize='small')
    if not os.path.exists('./plots/predictions'):
        os.makedirs('./plots/predictions')
    plt.savefig('./plots/predictions/Predictions.pdf')
    plt.close(fig)
    return parms_best


#Create the model object
model = models.MyModel()

#model.run(model.pdef)
#Generate synthetic observations
model.load_forcings('US-Ha1')
synthetic_err = 1.0
model.generate_synthetic_obs(model.pdef,synthetic_err)
#model.load_obs() #if you want to use real observations, uncomment this line

#Get the best parameters
pbound = numpy.stack([model.pmin, model.pmax]).transpose()   #prepare bounds for diff. evolution
result = differential_evolution(posterior_min, pbound, maxiter=100, popsize=100, disp=True)
print result
stop
#parms = result.x
#parms = QPSO(100, 100)
#stop
#model.run(parms)
res = minimize(posterior_min, model.pdef, method='Nelder-Mead', \
                 options={'maxiter': 3000, 'xtol': 1e-6, 'disp': True})
print res
parms=res.x
stop

parms = MCMC(model.pdef, 12000, burnsteps=5, nburn=2000)
levels = [-300, 0, 300, 600, 900, 1200, 1500, 1800, 2100, 2400, 2700,3000,3300]
f1 = plt.figure()
CS = plt.contour(model.output*24*3600*365,15,linewidths=0.2,colors='k')
CS = plt.contourf(model.output*24*3600*365,15,cmap=plt.cm.jet,levels=levels)
plt.colorbar() # draw colorbar
#for i in range(0,model.nsites):
#    plt.plot(parms[i*2], parms[i*2+1], 'kx')

#show observaitons
f2=plt.figure()
CS2 = plt.contour(model.obs*24*3600*365,15,linewidths=0.2,colors='k')
CS2 = plt.contourf(model.obs*24*3600*365,15,cmap=plt.cm.jet,levels=levels)
plt.colorbar() # draw colorbar
#for i in range(0,model.nsites):
#    plt.plot(parms[i*2], parms[i*2+1], 'kx')

plt.show()
#Run MCMC

