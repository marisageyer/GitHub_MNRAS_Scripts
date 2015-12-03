#!/usr/bin/python

import pypsr_redo as psr
#import scatteringmodels as sm

import os, sys
import matplotlib.pyplot as plt
from lmfit import Model, conf_interval, printfuncs
from math import pi
from lmfit import minimize, Parameter, Parameters, fit_report
import numpy as np
from scipy import special
import scipy.ndimage.filters as snf
import time

#### MODEL 1 #####################
def GxETrain(x,mu,sigma, A, tau):
    #This model convolves a pulsetrain of length 20 with a broadening function defined over several P's (lf*P)
    #It extracts one of the last convolved profiles, subtracts the climbed baseline and then adds noise to it
    bins, profile = psr.makeprofile(nbins = P, ncomps = 1, amps = A, means = mu, sigmas = sigma)
    binstau = np.linspace(1,P,P)  #Tested: having a longer exp here makes no difference
    scat = psr.psrscatter(psr.broadfunc(binstau,tau),psr.pulsetrain(3, bins, profile))   
    climb, observed_nonoise, rec, flux = psr.extractpulse(scat, 2, P)
    return observed_nonoise
#    return climb
    
#### MODEL 2 #####################
def GxESingleFold(x,mu,sigma,A,tau,trainlength):
    #This model takes a single Guassian pulse with mean mu and sigma
    #Convolves it with a broadening function
    #It extracts one of the last convolved profiles subtracts the climbed baseline and then adds noise to it 
    observed_postfold = np.zeros(P)      
    bins, profile = psr.makeprofile(nbins = P, ncomps = 1, amps = A, means = mu, sigmas = sigma)
    binstau = np.linspace(1,trainlength*P,trainlength*P)
    scat = psr.psrscatterpostfold(psr.broadfunc(binstau,tau),psr.pulsetrain(1, bins, profile))
    climb, observed_nonoise, rec, flux = psr.extractpulse(scat, 0, trainlength*P)
    for i in range(trainlength*P):    
        observed_postfold[np.mod(i,P)] += observed_nonoise[i]         
        GxESingleFold = observed_postfold[x]-np.min(observed_postfold[0:P])
    return GxESingleFold  
    
    
def EMG(x,mu,sigma,A,tau):
    return (((1/tau)/2)*A*np.exp(((1/tau)/2)*(2*mu+(1/tau)*sigma**2-2*x))*special.erfc((mu+(1/tau)*sigma**2-x)/(np.sqrt(2)*sigma)))
    
def PLM(x,K,k):
    return K*pow(x,-k)


## Set up timer
starttime = time.time()

timestr = time.strftime("%Y%m%d-%H")
print timestr
folderstr = 'Aniso'

### Universal constants 
light = 9.71561189*10**-12      #Speed of light in kpc/s  
mrad = 4.85*10**-9              #Convertion from miliarcseconds to radians

### Define global parameters
global nbinspow  
global P
global trainlength

nbinspow = 9
P = 2**nbinspow   #### note that this is independently put into the functions. Make sure they match! 
print 'bins: '+str(P)



### Input the observed frequency range

bandw = float(50)/float(1000) #GHz
#cfreq = float(150)/float(1000) # center freq GHz
incr = 0.005
#incr = 0.002

nulow, nuhigh = float(0.06),float(0.125)

nurange = np.arange(nulow,nuhigh,incr)

print 'bandwidth: '+str(bandw)+' GHz'
print 'nurange:'+str(nulow)+'-'+str(nuhigh)+' GHz'

#sys.exit()


### Create the properties of the ingoing pulse

pulseperiod =1.0                 #in seconds
dutycycle = float(2.5)             #as a % of the overall pulseperiod
m = float(P/4)                      #Let the mean of the pulse be at a 1/4 overall bins
w50 = float((dutycycle/100)*P)      #FWHM
s = w50/(2*np.sqrt(2*np.log(2)))    #sigma calculated through the dutycycle
a = 1                               #amplitude. at some point this will be replaced by the intrinsic spectrum. 

#Choose the trainlength to represent a time length of 8 times the tau value at the given frequency

bin2sec = pulseperiod/P
bs= bin2sec

## Intrinsic profile

bins, profile = psr.makeprofile(nbins = P, ncomps = 1, amps = a, means = m, sigmas = s)
#plt.figure()
#plt.plot(bins,profile)
#plt.figure()
#plt.plot(bins*bs,profile*bs)


#xaxlong =  np.linspace(1,100*P,100*P)

spectralindex = 1.6  #Input spectral index as a postivie number, for nu^-alpha
profile_intr = psr.profilespec(nurange,spectralindex,profile)
profile_intr_norm = profile_intr/np.sum(profile_intr[0])   #Normalised such that the max intrinsic flux = 1.0. That is the intrinsice pulse at the lowest frequency has flux = 1

#sys.exit()

###############################################################################
## SCATTERING SETUP
###############################################################################

Dval, Dsval = float(3.0),float(1.5)

#for k1 in range(3,15,3):
k1 = 3
k2 = 1
kappa1 = k1*mrad
kappa2 = k2*mrad

tauval = psr.tau(Dval,Dsval,kappa1,nurange,light)
tauval2 = psr.tau(Dval,Dsval,kappa2,nurange,light)
tauvalgeo = np.sqrt(tauval*tauval2)

#print tauval
#print tauval2
#print tauvalgeo

taubins = (tauval/pulseperiod)*P
taubins2 = (tauval2/pulseperiod)*P
taubinsgeo = np.sqrt(taubins*taubins2)

#print taubins
#sys.exit()

print '=============================='    
xax = np.arange(0,P,1)
    
    
### The models created above (GxFE and GxE) respectively fit for a convolution of 2 Gaussian pulses with a FOLDED and normal exponential. The code will show that GxFE performs best.
##
###Create limits on s, via w50
###The distribution of pulsar duty cylces is heavily skewed with a median at 2.5% and an overall minimum at 0.3% and overall maximum at 63% (Ref:Jayanth)
###This max is clearly huge - and therefore the process is pretty much unconstrained. Should consider inserting the actual distribution
##Create observed pulses by scattering a pulsetrain with an exponential broadening function


## All the combo arrays will have size (numruns,len(nurange),P)
## These contain the simulated profiles and fits for a given frequency and a given iteration
observedcombo= []
resultcombo = []
resultcheckcombo=[]
observedlowpasscombo=[]

## All the list arrays have size (numruns,len(nurange))
## These contain the fitted tauvalues and their standard deviations for a given frequency and iteration
obtainedtaulist = []
obtainedtaustdlist = []
obtainedtauchecklist = []
obtainedtaucheckstdlist = []

## All the alpha arrays have size(numruns)
## These contain the spectral indices of a given iteration
alpha1 = []
alphacheck = []

resultpows = []
resultpowscheck = []


snr = 20
print 'snr: '+str(snr)

numruns = 100
for count in range(numruns):
 
    observed = []
    observednonoise = []
    observedflux = []
    scatt = []
    checkfloors = []
    #observedfilter=[]
    resmedminarray=[]
    percarray=[]
    reslpminarray=[]
    observedlowpass=[]
        
    results = []
    resultscheck = []
          
    obtainedtau=[]
    obtainedtaustd=[]     
#    obtainedtau2=[]
#    obtainedtaustd2=[]    
    obtainedchecktau = []
    obtainedchecktaustd = []
    
    #        plt.figure(1)
        
    for k in range(0,len(taubins)):
        trainlength = int(10*taubins[k]/P)
        if trainlength < 5:
            trainlength = 5
        print ''
        print '%.3f GHz' % nurange[k]
        print 'trainlength: '+str(trainlength)   
#        scat = psr.psrscatter(psr.broadfunc(xaxlong,taubins[k]),psr.pulsetrain(trainlength, bins, profile_intr_norm[k]))
        scat = psr.psrscatter(psr.broadfunc2(xax,taubins[k],taubins2[k]),psr.pulsetrain(3, bins, profile_intr_norm[k]))        
        scatt.append(scat)        
        climb, observed_nonoise, rec, flux = psr.extractpulse(scat, 2, P)
        
        peak = np.max(observed_nonoise)
        noise = np.random.normal(0,peak/snr,P)
        observedadd = climb + noise
        checkfloor = observed_nonoise + noise
        observed.append(observedadd)
        
        observedflux.append(flux)
        observednonoise.append(observed_nonoise)
        checkfloors.append(checkfloor)
        
        ## Method 1: Create a median filter around the absolute minimum, to find minimum that represents the underlying noiseless minimum well.    
#        idmin = np.argmin(observedadd)
#        filterwindow = 60
#        fstart = idmin-filterwindow
#        if fstart < 0:
#            fstart = 0
#        fend = idmin+filterwindow
#        if fend > P:
#            fend = P-1
#        print fstart,fend
#        filterrange = observedadd[fstart:fend]
#        samp = 40
#        mediansampling = P/samp
#        medfilter = snf.median_filter(filterrange, size=mediansampling)
#        filmin = np.min(medfilter)
#        observedzeromed = observedadd - filmin
#        resmedmin = np.min(climb) - filmin
#        percmedmin = np.abs(resmedmin/np.min(climb)*100)
#        resmedminarray.append(resmedmin)
#    
#        plt.figure((count+1)*100000+k)
#        plt.plot(climb,'r')
#        plt.plot(medfilter,'b')
#    
        ## Method 2: low Pass Filter
        edge = 20
        fourier = np.fft.fft(observedadd)
        fouriersubset = np.concatenate((fourier[0:edge],fourier[P-edge:P]))
        invfourier = np.fft.ifft(fouriersubset)
        binfix = np.linspace(1,P,len(invfourier))
        scalefix = len(invfourier)/float(P)
        lowpasssignal = np.real(scalefix*invfourier)
        lowpassmin = np.min(lowpasssignal)
        if taubinsgeo[k]/P < 0.06:
            lowpassmin = np.mean(observedadd[0:70])
            print "No run-in. Min: %.2e" %lowpassmin
        else:
            lowpassmin = lowpassmin
            print "Run-in. Min: %.2e" %lowpassmin
        observedzerolowpass = observedadd - lowpassmin
        observedlowpass.append(observedzerolowpass)
        reslpmin = np.min(climb) - lowpassmin
        perclpmin = np.abs(reslpmin/np.min(climb)*100)
    #    reslpminarray.append(reslpmin)
    #  
#        plt.figure(200+k)
#        plt.plot(scatt[k],'r', linewidth=2.0)
#        plt.title('Has the climb stabilised?, %.0f MHz' % (1000*nurange[k]),fontsize=16)
#    
#        plt.figure((count+1)*10000+k)
#        plt.plot(binfix,lowpasssignal,'r--',linewidth = 2.0)
#        plt.plot(observedadd,'g--')
#        plt.plot(climb, 'k')
#        plt.title('%.3f GHz, edge: %d' % (nurange[k],edge))
#        
#        plt.figure((count+1)*100000+k)
#        plt.plot(observedzerolowpass,'m--')
#        plt.plot(checkfloor,'c--')
#        plt.plot(observed_nonoise, 'k')
#        plt.title('%.3f GHz, edge: %d' % (nurange[k],edge)) 

    
        ## Set the parameters for the different models. They have become frequency dependent since trainlength is frequency dependent.
        modelname = GxETrain
        model = Model(modelname)
        
        model.set_param_hint('sigma', value=s, vary=True)
        model.set_param_hint('mu', value=m, vary=True)
        model.set_param_hint('A',value=1.5, vary=True, min=0)
        model.set_param_hint('tau',value=200, vary=True, min=0)
        pars = model.make_params()
        
#        modelname2 = GxESingleFold
#        model2 = Model(modelname2)
#        
#        model2.set_param_hint('sigma', value=s, vary=True)
#        model2.set_param_hint('mu', value=m, vary=True)
#        model2.set_param_hint('A',value=1.5, vary=True, min=0)
#        model2.set_param_hint('tau',value=200, vary=True, min=0)
#        model2.set_param_hint('trainlength',value=trainlength, vary=False)
#        pars2 = model2.make_params()
        
        modelpow = Model(PLM)
        
        xsec = np.linspace(0,pulseperiod,P)
    
#        sp = 4 #number of subplots per figure 
#        numFigs = k/4 + 1
#        totFig = int(len(taubins)/sp) + 1
#        print "Figure nr: %d/%d" % (numFigs,totFig)
#        plt.figure(count*numruns + numFigs, figsize=(14,9))
#        subplotcount = (k+1) - sp*(numFigs-1)
#        plt.subplot(2,2,subplotcount)
#        plt.subplots_adjust(hspace=.3)             
#        plt.plot(xsec,observedlowpass[k]/np.max(checkfloors[k]), 'g-', alpha=0.7)
      
      
    ## Fit Model 1:                    
        result = model.fit(observedlowpass[k],pars,x=xax)
    #   print(result.fit_report(show_correl = False))
        besttau = result.best_values['tau']
        taustd = result.params['tau'].stderr
        obtainedtau.append(besttau)  
        obtainedtaustd.append(taustd)
        results.append(result)
               
#        plt.plot(xsec,result.best_fit/np.max(checkfloors[k]),'r-', linewidth = 2.0,label = r'%s $\tau=%.4f$' % (r'Single $\tau$: pre-fold', obtainedtau[k]*bs))     
    
    ## Fit Model 2:    
#        result2 = model2.fit(observedlowpass[k],pars2,x=xax)
#        besttau2 = result2.best_values['tau']
#        taustd2 = result2.params['tau'].stderr              
#        obtainedtau2.append(besttau2)
#        obtainedtaustd2.append(taustd2)
#        
#        plt.plot(xsec,result2.best_fit/np.max(observedlowpass[k]),'b--',linewidth = 2.0, label = r'%s, $\tau=%.4f$' % ('Post-fold', obtainedtau2[k]*bs)) 
    
    
    ## Fit Check with Model 1:        
        resultcheck = model.fit(checkfloors[k],pars,x=xax)                    
        checktau = resultcheck.best_values['tau']
        checktaustd = resultcheck.params['tau'].stderr
        obtainedchecktau.append(checktau)
        obtainedchecktaustd.append(checktaustd)
        resultscheck.append(resultcheck)
        
#        plt.plot(xsec,resultcheck.best_fit/np.max(checkfloors[k]),'c--',linewidth = 2.0, label = r'%s, $\tau=%.4f$' % (r'Single $\tau$: check' , obtainedchecktau[k]*bs)) 

#        plt.title(r'%.0f MHz Input: $\tau1=%.3fs$, $\tau2 = %.3fs$, $\tau_{geo} = %.3fs$' % (1000*nurange[k],tauval[k],tauval2[k],tauvalgeo[k]), fontsize=12)
#        plt.title(r'%.0f MHz, simulated $\tau=%.4f$, ' % (1000*nurange[k],tauval[k]),fontsize=12) 
#        plt.legend(loc = 'upper right',fontsize=11)
#        plt.xlabel('time (sec)')
#        plt.xlim(0,pulseperiod)
        k += 1
        
        
    #    filename = 'IsoProfiles_'+str(j)+'.png'
    #    picpath = "/Users/marisa/Documents/PhD/GitHub_Scattering/Plots"
    #    fileoutput = os.path.join(picpath,filename)
    #    plt.savefig(fileoutput, dpi=96)
    #
    obtainedtau = np.array(obtainedtau) 
#    obtainedtau2 = np.array(obtainedtau2) 
    obtainedchecktau = np.array(obtainedchecktau)
       
    obtainedtaustd = np.array(obtainedtaustd)
#    obtainedtaustd2 = np.array(obtainedtaustd2)
    obtainedchecktaustd = np.array(obtainedchecktaustd)
    
    #        bwhigh2 = nurange+0.02/2
    #        bwlow2 = nurange-0.02/2
    #        rfreq2 = 10**((np.log10(bwhigh2)+ np.log10(bwlow2))/2)                
    #        resultpowrf = modelpow.fit(obtainedtau,x=rfreq,K=0.001,k=4)
    #        resultpowrf2 = modelpow.fit(obtainedtau,x=rfreq2,K=0.001,k=4)        
    #        specfitrf = resultpowrf.best_values['k']
    #        specfitrf2 = resultpowrf2.best_values['k']
    
    resultpow = modelpow.fit(obtainedtau,x=nurange,K=0.001,k=4)        
    specfit = resultpow.best_values['k']
    print 'alpha_1 = %.4f' %specfit
     
#    resultpow2 = modelpow.fit(obtainedtau2,x=nurange,K=0.001,k=4)
#    specfit2 = resultpow2.best_values['k']
#    print 'alpha_2 = %.4f' %specfit2    
#       
    resultpowcheck = modelpow.fit(obtainedchecktau,x=nurange,K=0.001,k=4)
    specfitcheck = resultpowcheck.best_values['k']
    print 'alpha_ch = %.4f' %specfitcheck                
    
    taupow4 = psr.tau(Dval,Dsval,kappa1,nurange,light)
    taupow4bins = taupow4*P/pulseperiod
    taupowAni4 = np.sqrt(taubins*taubins2)
    
    # Frequency range as determined by the chosen bandwidth
    bwhigh = nurange+bandw/2
    bwlow = nurange-bandw/2
    
    rfreq = 10**((np.log10(bwhigh)+ np.log10(bwlow))/2)*1000
    nurangeMHz = nurange*1000
    
    bin2sec = pulseperiod/P
    bs= bin2sec
       
#    ticksMHz = []
#    for i in range(0,len(nurange),1):
#        tMHz = int(round(nurange[i]*1000))
#        ticksMHz.append(tMHz)
#    ticksMHz.append(int(round(1000*nurange[len(nurange)-1])))
#    ticksMHz = ticksMHz[0:len(ticksMHz):2]
##    
#    plt.figure((count+1)*100000, figsize=(14, 9))
#    plt.errorbar(nurange*1000,obtainedtau*bs,yerr=obtainedtaustd*bs,fmt='ro',markersize=6, capthick=2,linewidth=1.5, label = r'%s: $\alpha=$ %.3f' % ('Pre-fold' , specfit))
##    plt.errorbar(nurange*1000,obtainedtau2*bs,yerr=obtainedtaustd2*bs,fmt='b^',markersize=10,capthick=2, linewidth = 1.5,label = r'%s, $\alpha=$ %.3f' % ('Post-fold, long', specfit2))
#    plt.errorbar(nurange*1000,obtainedchecktau*bs,yerr=obtainedchecktaustd*bs,fmt='co',markersize=8,capthick=2, linewidth = 1.5,label = r'%s, $\alpha=$ %.3f' % ('Check', specfitcheck))
#    plt.plot(nurange*1000,resultpow.best_fit*bs,'r--',linewidth = 2.0)
##    plt.plot(nurange*1000,resultpow2.best_fit*bs,'b--',linewidth = 2.0)
#    plt.plot(nurange*1000,resultpowcheck.best_fit*bs,'c--',linewidth = 2.0)
#    #        plt.errorbar(rfreq*1000,obtainedtau*bs,yerr=obtainedtaustd*bs,fmt='k^',markersize=10, capthick=2,linewidth=1.5, label = r'%s: $\alpha=$ %.3f, $\Delta f$= 50MHz' % (r'$f_c$' , specfitrf))        
#    #        plt.plot(rfreq*1000,resultpowrf.best_fit*bs,'k-',linewidth = 2.0)
#    #        plt.plot(rfreq2*1000,resultpowrf.best_fit*bs,'c-',linewidth = 2.0)
#    #        plt.errorbar(rfreq2*1000,obtainedtau*bs,yerr=obtainedtaustd*bs,fmt='c*',markersize=10, capthick=2,linewidth=1.5, label = r'%s: $\alpha=$ %.3f, $\Delta f$= 20MHz' % (r'$f_c$' , specfitrf2))               
#    #        plt.plot(rfreq*1000,taubins*bs,'c*-',markersize=8.0,linewidth=1.5,label = r'$\alpha= 4.00$')        
#    #for i in range(len(taubins)):        
#    #    plt.text(1000*nurange[i],1.1*bs*obtainedtau[i],'%.1f' % percarray[i])        
#    plt.plot(nurange*1000,taupowAni4*bs,'m*-',markersize=8.0,linewidth=1.5,label = r'$\alpha= 4.00$')        
#    plt.xlim(nurange[0]*1000-5,nuhigh*1000+5)
##    plt.xscale('log')
#    #plt.yscale('log')
#    plt.xticks(ticksMHz,ticksMHz,fontsize=20)
#    plt.yticks(fontsize=20)
#    plt.xlabel('frequency (MHz)', fontsize=26)
#    plt.ylabel(r'$\tau$ (sec)',fontsize=26)
#    #plt.legend(loc ='lower left',fontsize=18)
#    plt.legend(loc ='best',fontsize=18)
#    
#
#    plt.figure((count+1)*1000,figsize=(14, 9))
#    plt.errorbar(nurange*1000,obtainedtau*bs,yerr=obtainedtaustd*bs,fmt='ro',markersize=6, capthick=2,linewidth=1.5, label = r'%s: $\alpha=$ %.3f' % (r'Single $\tau$: pre-fold' , specfit))
##    plt.errorbar(nurange*1000,obtainedtau2*bs,yerr=obtainedtaustd2*bs,fmt='b^',markersize=10,capthick=2, linewidth = 1.5,label = r'%s, $\alpha=$ %.3f' % ('Post-fold, long', specfit2))
#    plt.errorbar(nurange*1000,obtainedchecktau*bs,yerr=obtainedchecktaustd*bs,fmt='co',markersize=6,capthick=2, linewidth = 1.5,label = r'%s, $\alpha=$ %.3f' % (r'Single $\tau$: check', specfitcheck))
#    plt.plot(nurange*1000,resultpow.best_fit*bs,'r--',linewidth = 2.0)
##    plt.plot(nurange*1000,resultpow2.best_fit*bs,'b--',linewidth = 2.0)
#    plt.plot(nurange*1000,resultpowcheck.best_fit*bs,'c--',linewidth = 2.0)  
##    plt.plot(nurange*1000,taubins*bs,'m*-',markersize=8.0,linewidth=1.5,label = r'$\alpha= 4.00$')        
### Ani:    
#    plt.plot(nurange*1000,taupowAni4*bs,'m*-',markersize=10.0,linewidth=1.5,label = r'$\alpha= 4.00$')
#    plt.plot(nurange*1000,taubins*bs,'g-',label=r'$\tau_1$ input')
#    plt.plot(nurange*1000,taubins2*bs,'g--',label=r'$\tau_2$ input')
#    plt.xlim(nurange[0]*1000-5,nuhigh*1000+5)
#    plt.xscale('log')
#    plt.yscale('log')
#    plt.xticks(ticksMHz,ticksMHz,fontsize=20)
#    plt.yticks(fontsize=20)
#    plt.xlabel('frequency (MHz)', fontsize=26)
#    plt.ylabel(r'$\tau$ (sec)',fontsize=26)
#    #plt.legend(loc ='lower left',fontsize=18)
#    plt.legend(loc ='best',fontsize=18)

    alpha1.append(specfit)
    alphacheck.append(specfitcheck)
    
    resultpows.append(resultpow)
    resultpowscheck.append(resultpowcheck)
    
    observedcombo.append(observed)
    resultcombo.append(results)
    resultcheckcombo.append(resultscheck)
    observedlowpasscombo.append(observedlowpass)
  
    obtainedtaulist.append(obtainedtau)
    obtainedtaustdlist.append(obtainedtaustd)
    obtainedtauchecklist.append(obtainedchecktau)
    obtainedtaucheckstdlist.append(obtainedchecktaustd)

    
#print alpha1
print np.mean(alpha1)
print np.std(alpha1)

#print alpha2
#print np.mean(alpha2)
#print np.std(alpha2)

#print alphacheck
print np.mean(alphacheck)
print np.std(alphacheck)
print ''    
print 'snr: '+str(snr)
print 'pulseperiod: '+str(pulseperiod)
print 'dutycycle: '+str(dutycycle)
print 'nurange:'+str(nulow)+'-'+str(nuhigh)+' GHz'+'('+str(incr)+')'
print 'edge:'+str(edge)
    
print"****************************************"
print"****************************************"


np.savetxt('Pythondata/taulists/%s/alpha1_%s.txt' %(folderstr,timestr), alpha1)
np.savetxt('Pythondata/taulists/%s/alphacheck_%s.txt' %(folderstr,timestr), alphacheck) 
   
  ## Find the numrun iteration that represents the mean of all the runs best
    
res1 = np.abs(np.mean(alpha1)-alpha1)
resch= np.abs(np.mean(alphacheck)-alphacheck)


## Choose the set of alpha's with least overall difference from mean
## I.e the set that represents the mean best

"""Use weighted mean for improved accuracy"""

weightval = np.linspace((1./numruns),1,numruns)
weights = weightval/np.sum(weightval)

indd = np.arange(0,numruns,1)

stack1 = np.column_stack((indd,res1))
stackch = np.column_stack((indd,resch))

sort1 = stack1[stack1[:,1].argsort()]
sort1_w = np.column_stack((sort1[:,0],weights[::-1])) ##::-1 just gives the reverese order of the weights, as I want the index associated with the smallest residual to have the largest weight
sort1_ind = sort1_w[sort1_w[:,0].argsort()]
#sort1[:,1] *= weights

sortch = stackch[stackch[:,1].argsort()]
sortch_w = np.column_stack((sortch[:,0],weights[::-1]))
sortch_ind = sortch_w[sortch_w[:,0].argsort()]

ind = np.argmax(sortch_ind[:,1]*sort1_ind[:,1])


## Plot tau spectrum


ticksMHz = []
for i in range(0,len(nurange),1):
    tMHz = int(round(nurange[i]*1000))
    ticksMHz.append(tMHz)
ticksMHz.append(int(round(1000*nurange[len(nurange)-1])))
ticksMHz = ticksMHz[0:len(ticksMHz):2]


plt.figure((count+1),figsize=(14, 9))
plt.errorbar(nurange*1000,obtainedtaulist[ind]*bs,yerr=obtainedtaustdlist[ind]*bs,fmt='ro',markersize=8, capthick=2,linewidth=1.5, label = r'%s: $\alpha=$ %.3f' % ('Pre-fold' , alpha1[ind]))
plt.errorbar(nurange*1000,obtainedtauchecklist[ind]*bs,yerr=obtainedtaucheckstdlist[ind]*bs,fmt='co',markersize=8,capthick=2, linewidth = 1.5,label = r'%s, $\alpha=$ %.3f' % ('Pre-fold check', alphacheck[ind]))
plt.plot(nurange*1000,resultpows[ind].best_fit*bs,'r--',linewidth = 2.0)
plt.plot(nurange*1000,resultpowscheck[ind].best_fit*bs,'c--',linewidth = 2.0)     
plt.plot(nurange*1000,taupowAni4*bs,'m*-',markersize=10.0,linewidth=1.5,label = r'$\alpha= 4.00$')
plt.plot(nurange*1000,taubins*bs,'g-',label=r'$\tau_1$ input')
plt.plot(nurange*1000,taubins2*bs,'g--',label=r'$\tau_2$ input')
plt.xlim(nurange[0]*1000-5,nuhigh*1000+5)
plt.xscale('log')
plt.yscale('log')
plt.xticks(ticksMHz,ticksMHz,fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('frequency (MHz)', fontsize=26)
plt.ylabel(r'$\tau$ (sec)',fontsize=26)
plt.legend(loc ='best',fontsize=22)
filename = 'NewTestTauspectrum'+str(nulow)+'_'+str(nuhigh)+'_log.png'
picpath = "/Users/marisa/Documents/PhD/GitHub_Scattering/Plots/%s" %folderstr
fileoutput = os.path.join(picpath,filename)
plt.savefig(fileoutput, dpi=150)



plt.figure((count+2),figsize=(14, 9))
plt.errorbar(nurange*1000,obtainedtaulist[ind]*bs,yerr=obtainedtaustdlist[ind]*bs,fmt='ro',markersize=8, capthick=2,linewidth=1.5, label = r'%s: $\alpha=$ %.3f' % ('Pre-fold' , alpha1[ind]))
plt.errorbar(nurange*1000,obtainedtauchecklist[ind]*bs,yerr=obtainedtaucheckstdlist[ind]*bs,fmt='co',markersize=8,capthick=2, linewidth = 1.5,label = r'%s, $\alpha=$ %.3f' % ('Pre-fold check', alphacheck[ind]))
plt.plot(nurange*1000,resultpows[ind].best_fit*bs,'r--',linewidth = 2.0)
plt.plot(nurange*1000,resultpowscheck[ind].best_fit*bs,'c--',linewidth = 2.0)     
plt.plot(nurange*1000,taupowAni4*bs,'m*-',markersize=10.0,linewidth=1.5,label = r'$\alpha= 4.00$')
plt.plot(nurange*1000,taubins*bs,'g-',label=r'$\tau_1$ input')
plt.plot(nurange*1000,taubins2*bs,'g--',label=r'$\tau_2$ input')
plt.xlim(nurange[0]*1000-5,nuhigh*1000+5)
plt.xticks(ticksMHz,ticksMHz,fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('frequency (MHz)', fontsize=26)
plt.ylabel(r'$\tau$ (sec)',fontsize=26)
#plt.legend(loc ='lower left',fontsize=18)
plt.legend(loc ='best',fontsize=22)
filename = 'NewTestTauspectrum'+str(nulow)+'_'+str(nuhigh)+'.png'
picpath = "/Users/marisa/Documents/PhD/GitHub_Scattering/Plots/%s" %folderstr
fileoutput = os.path.join(picpath,filename)
plt.savefig(fileoutput, dpi=150)


## Arrays to be saved:
tau1 = obtainedtaulist[ind]
tauch = obtainedtauchecklist[ind]

tau1std = obtainedtaustdlist[ind]
tauchstd = obtainedtaucheckstdlist[ind]

np.savetxt('Pythondata/taulists/%s/tau1_%s.txt' %(folderstr,timestr), tau1)
np.savetxt('Pythondata/taulists/%s/tauch_%s.txt' %(folderstr,timestr), tauch) 

np.savetxt('Pythondata/taulists/%s/tau1std_%s.txt' %(folderstr,timestr), tau1std) 
np.savetxt('Pythondata/taulists/%s/tauchstd_%s.txt' %(folderstr,timestr), tauchstd)  
  

## Plot profiles
sp = 4 #number of subplots per figure 

for k in range(0,len(taubins)):
    numFig = k/4 + 1
    totFig = int(len(taubins)/sp) + 1
    print "Figure nr: %d/%d" % (numFig,totFig)
    plt.figure(count*numruns + numFig, figsize=(17,10))
    subplotcount = (k+1) - sp*(numFig-1)
    plt.subplot(2,2,subplotcount)
    plt.subplots_adjust(hspace=.4)           
    plt.plot(xsec,observedlowpasscombo[ind][k]/np.max(observednonoise[k]),'g-',alpha=0.7)
    plt.plot(xsec,resultcombo[ind][k].best_fit/np.max(observednonoise[k]),'r-', linewidth = 2.0,label = r'%s $\tau=%.4f \pm %.4f$' % ('Pre-fold', obtainedtaulist[ind][k]*bs,obtainedtaustdlist[ind][k]*bs))     
    plt.plot(xsec,resultcheckcombo[ind][k].best_fit/np.max(observednonoise[k]),'c--',linewidth = 2.0, label = r'%s, $\tau=%.4f \pm %.4f$' % ('Pre-fold check' ,obtainedtauchecklist[ind][k]*bs,obtainedtaucheckstdlist[ind][k]*bs)) 
    plt.title(r'%.0f MHz Input: $\tau1=%.3fs$, $\tau2 = %.3fs$, $\tau_{geo} = %.3fs$' % (1000*nurange[k],tauval[k],tauval2[k],tauvalgeo[k]), fontsize=20)
    plt.legend(loc = 'upper right',fontsize=13)
    plt.xlabel('time (sec)',fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlim(0,pulseperiod)

    filename = 'NewTestProfiles'+str(numFig)+'.png'
    picpath = "/Users/marisa/Documents/PhD/GitHub_Scattering/Plots/%s" %folderstr
    fileoutput = os.path.join(picpath,filename)
    plt.savefig(fileoutput, dpi=150)   

## Arrays to be saved:
    arNoN = observednonoise[k]
    arN = observedlowpasscombo[ind][k]
    ar1 = resultcombo[ind][k].best_fit
    arch = resultcheckcombo[ind][k].best_fit 

    np.savetxt('Pythondata/arNoN/%s/arNoN_%d_%s.txt' %(folderstr,k,timestr), arNoN)
    np.savetxt('Pythondata/arN/%s/arN_%d_%s.txt' %(folderstr,k,timestr), arN)
    np.savetxt('Pythondata/ar1/%s/ar1_%d_%s.txt' %(folderstr,k,timestr), ar1)
    np.savetxt('Pythondata/arch/%s/arch_%d_%s.txt' %(folderstr,k,timestr), arch)
    
arsec = xsec
np.savetxt('Pythondata/arsec/arsec_Ani.txt', arsec)
  


#print resminarray        


#        percmin = np.abs(resminarray/np.min(climb)*100)
#        perctaures = np.abs(np.divide((obtainedchecktau - obtainedtau),obtainedchecktau))*100

#        combinedstd = np.sqrt(np.power(obtainedtaustd,2) + np.power(obtainedchecktaustd,2))
#        errorbar = np.divide(combinedstd,obtainedchecktau)*100







#plt.figure(figsize=(14, 9))
#plt.errorbar(nurange*1000,obtainedtau*bs,yerr=obtainedtaustd*bs,fmt='ro',markersize=10, capthick=2,linewidth=1.5, label = r'%s: $\alpha=$ %.3f' % (r'$f_m$' , specfit))
#plt.errorbar(nurange*1000,obtainedtau2*bs,yerr=obtainedtaustd2*bs,fmt='b^',markersize=10,capthick=2, linewidth = 1.5,label = r'%s, $\alpha=$ %.3f' % ('Post-fold, long', specfit2))
#plt.errorbar(nurange*1000,obtainedtau3*bs,yerr=obtainedtaustd3*bs,fmt='g*',markersize=10,capthick=2, linewidth = 1.5,label = r'%s, $\alpha=$ %.3f' % ('Folded Exponential', specfit3))
#plt.errorbar(nurange*1000,obtainedchecktau*bs,yerr=obtainedchecktaustd*bs,fmt='co',markersize=10,capthick=2, linewidth = 1.5,label = r'%s, $\alpha=$ %.3f' % ('Check', specfitcheck))       
#plt.plot(nurange*1000,resultpow.best_fit*bs,'r--',linewidth = 2.0)
#plt.plot(nurange*1000,resultpow2.best_fit*bs,'b--',linewidth = 2.0)
#plt.plot(nurange*1000,resultpow3.best_fit*bs,'g--',linewidth = 2.0)
#plt.plot(nurange*1000,resultpowcheck.best_fit*bs,'c--',linewidth = 2.0)  
#plt.plot(nurange*1000,taubins*bs,'m*-',markersize=8.0,linewidth=1.5,label = r'$\alpha= 4.00$')        
#plt.xlim(nulow*1000,nuhigh*1000)
#plt.xscale('log')
#plt.yscale('log')
#plt.xticks(ticksMHz,ticksMHz,fontsize=20)
#plt.yticks(fontsize=20)
#plt.xlabel('frequency (MHz)', fontsize=26)
#plt.ylabel(r'$\tau$ (sec)',fontsize=26)
#plt.legend(loc ='lower left',fontsize=18)
