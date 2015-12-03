#!/usr/bin/python

"""
Simulate and fit Scattered pulse profiles

@author: marisa

Makes use of functions in pypsr_redo.py
"""

import pypsr_redo as psr
import os, sys
import matplotlib.pyplot as plt
from lmfit import Model, conf_interval, printfuncs
from math import pi
from lmfit import minimize, Parameter, Parameters, fit_report
import numpy as np
from scipy import special
import scipy.ndimage.filters as snf
import time

"""
Outline of this script:

1.  Definitions of fitting models: a pre-fold model GxETrain, a post-fold model GxESingleFold, and an exponentially modified Gaussian (EMG)
2.  Definition of power-law model to fit tau-spectra with
3.  Initiate some conversion constants, bin values, as well as the chosen observed frequency range and bandwidth. (The bandwidth will only be used to calculate the central frequency fc to which the fitted tau value corresponds. It's not part of the fitting process.)
4.  Define the intrinsic pulse
"""


"""
1. Models for scattered pulses
"""

#### MODEL 1 #####################
def GxETrain(x,mu,sigma, A, tau):
    #This model convolves a pulsetrain with a broadening function
    #It extracts one of the last convolved profiles, subtracts the climbed baseline and then adds noise to it
    bins, profile = psr.makeprofile(nbins = P, ncomps = 1, amps = A, means = mu, sigmas = sigma)
    binstau = np.linspace(1,P,P)
    scat = psr.psrscatter(psr.broadfunc(binstau,tau),psr.pulsetrain(3, bins, profile))   
#    plt.figure()
#    plt.plot(scat,'r')    
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
    
#### MODEL 3 #####################
def EMG(x,mu,sigma,A,tau):
    return (((1/tau)/2)*A*np.exp(((1/tau)/2)*(2*mu+(1/tau)*sigma**2-2*x))*special.erfc((mu+(1/tau)*sigma**2-x)/(np.sqrt(2)*sigma)))

"""
2. Power law fit
"""
def PLM(x,K,k):
    return K*pow(x,-k)



"""
3. Global parameters, conversions, bins, frequency
"""

## Set up timer
starttime = time.time()

### Define global parameters
global nbinspow  
global P
global trainlength

### Universal constants 
light = 9.71561189*10**-12      #Speed of light in kpc/s  
mrad = 4.85*10**-9              #Convertion from miliarcseconds to radians

### Define number of bins (this will typically represent one pulse period)
nbinspow = 9
P = 2**nbinspow

print 'bins: '+str(P)


### Input the observed frequency range

bandw = float(50)/float(1000) #GHz
incr = 0.01
#incr = 0.005

nulow, nuhigh = float(0.05),float(0.12)
nurange = np.arange(nulow,nuhigh,incr)
print 'bandwidth: '+str(bandw)+' GHz'
print 'nurange:'+str(nulow)+'-'+str(nuhigh)+' GHz'

#sys.exit()



"""
4. Intrinsic Pulse
"""

### Create the properties of the ingoing pulse
pulseperiod =1.0                   # in seconds
dutycycle = float(2.5)               # as a % of the overall pulseperiod
m = float(P/4)                       # Let the mean of the pulse be at a 1/4 overall bins
w50 = float((dutycycle/100)*P)       # FWHM
s = w50/(2*np.sqrt(2*np.log(2)))     # sigma calculated through the dutycycle
a = 1                                # amplitude. at some point this will be replaced by the intrinsic spectrum. 


bin2sec = pulseperiod/P              # convert between bins and pulseperiod (time)
bs= bin2sec

### Make Guassian profile

bins, profile = psr.makeprofile(nbins = P, ncomps = 1, amps = a, means = m, sigmas = s)
#plt.figure()
#plt.plot(bins,profile)
#plt.figure()
#plt.plot(bins*bs,profile*bs)

### Scale the flux of the profile according to spectral index

spectralindex = 1.6                 # Input spectral index as a postivie number, for nu^-alpha
profile_intr = psr.profilespec(nurange,spectralindex,profile)
profile_intr_norm = profile_intr/np.sum(profile_intr[0])   # Normalised such that the max intrinsic flux = 1.0. That is the intrinsice pulse at the lowest frequency has flux = 1

#sys.exit()

"""
5. Scattering Setup
"""
## Dval is the overall distance from pulsar to observer (in kpc)
## Dsval is the distance from the source to the screen  (in kpc)
Dval, Dsval = float(3.0),float(1.5)

## k1 in mas describes the proportionality in the cold plasma relation: sigma_a \propto nu^-2
## Cordes and Lazio 2001 suggest a 'typical' scattering strength of 3mas at 1 GHz.

#for k1 in range(3,15,3):
k1 = 3.0  #(in mas)
kappa1 = k1*mrad  #(convert to radians)

## Value of tau (in sec.) at a given freq.
## See e.g. eq. 17 in Cordes and Lazio 2001

#tauval = psr.tau(Dval,Dsval,kappa1,nurange,light)
tauval = psr.tauE(Dval,kappa1,nurange,light)
#print tauval

## Tau value converted from seconds to bins
taubins = (tauval/pulseperiod)*P
#print taubins
#print np.median(taubins)

#sys.exit()

print '=============================='    

"""
6. Initiate lists to append to
"""
#xax = np.arange(0,P,1)
xax = np.arange(1,P,1)

## All the combo arrays will have size (numruns,len(nurange),P)
## These contain the simulated profiles and fits for a given frequency and a given iteration
observedcombo= []
resultcombo = []
result2combo=[]
result3combo=[]
resultcheckcombo=[]
observedlowpasscombo=[]

## All the list arrays have size (numruns,len(nurange))
## These contain the fitted tauvalues and their standard deviations for a given frequency and iteration
obtainedtaulist = []
obtainedtaustdlist = []
obtainedtau2list = []
obtainedtau2stdlist = []
obtainedtau3list = []
obtainedtau3stdlist = []
obtainedtauchecklist = []
obtainedtaucheckstdlist = []

## All the alpha arrays have size(numruns)
## These contain the spectral indices of a given iteration
alpha1 = []
alpha2 = []
alpha3 = []
alphacheck = []

resultpows = []
resultpows2 = []
resultpows3 = []
resultpowscheck = []

## Choose a peak signal to noise ratio
snr = 20

## Choose a number of iterations to run.
## For each numrun the script will:
##  a) simulate noisy scatter broadened profiles at the frequency range specified in section 3.
##  b) fit the broadened profiles with the chosen models to obtain a characteristic scattering time value
##  c) plot tau-spectrum  

#timestr = 'HF'
#timestr = time.strftime("%Y%m%d-%H")
#print timestr
 
numruns = 4
for count in range(numruns):
    print '-------------------------------------------'
    print 'count: %d' % count
    
    observed = []
    observednonoise = []
    observedflux = []
    scatt = []
    checkfloors = []
    percarray=[]
    reslpminarray=[]
    observedlowpass=[]
        
    results = []
    results2 = []
    results3 = []
    resultscheck = []
    
    obtainedtau=[]
    obtainedtaustd=[]
     
    obtainedtau2=[]
    obtainedtaustd2=[]
    
    obtainedtau3=[]
    obtainedtaustd3=[]
        
    obtainedchecktau=[]
    obtainedchecktaustd=[]
 
    #        plt.figure(1)
        
    for k in range(0,len(taubins)):
        trainlength = int(round(20*taubins[k]/P))
        if trainlength < 5:
            trainlength = 5     
        print 'trainlength: '+str(trainlength)
        scat = psr.psrscatter(psr.broadfunc(xax,taubins[k]),psr.pulsetrain(3, bins, profile_intr_norm[k]))
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
#        observedzerolowpass = observedadd - filmin
#        resmedmin = np.min(climb) - filmin
#        percmedmin = np.abs(resmedmin/np.min(climb)*100)
#        observedlowpass.append(observedzerolowpass)
##        resmedminarray.append(resmedmin)
    
    #    plt.figure()
    #    plt.plot(climb,'r')
    #    plt.plot(medfilter,'b')
    
        ## Method 2: low Pass Filter
        edge = 20
        fourier = np.fft.fft(observedadd)
        fouriersubset = np.concatenate((fourier[0:edge],fourier[P-edge:P]))
        invfourier = np.fft.ifft(fouriersubset)
        binfix = np.linspace(1,P,len(invfourier))
        scalefix = len(invfourier)/float(P)
        lowpasssignal = np.real(scalefix*invfourier)
        lowpassmin = np.min(lowpasssignal)            
        if taubins[k]/P < 0.06:
            lowpassmin = np.mean(observedadd[0:70])
            print "No run-in. Min: %.2e" %lowpassmin
        else:
            lowpassmin = lowpassmin
            print "Run-in. Min: %.2e" %lowpassmin
        observedzerolowpass = observedadd - lowpassmin
        observedlowpass.append(observedzerolowpass)
#    #    reslpmin = np.min(climb) - lowpassmin
    #    perclpmin = np.abs(reslpmin/np.min(climb)*100)
    #    reslpminarray.append(reslpmin)
        
#        plt.figure(200+k)
#        plt.plot(scatt[k],'r', linewidth=2.0)
#        plt.title('Has the climb stabilised?, %.0f MHz' % (1000*nurange[k]),fontsize=16)
        
#        plt.figure()
#        plt.plot(binfix,lowpasssignal - lowpassmin, 'm--', linewidth=2.0,label = 'Low Pass Filter')        
#        plt.plot(observed_nonoise,'r',linewidth=2.0,label = 'No noise data')
#        plt.ylim(-0.0002)
#        plt.legend()
    
    
        ## Set the parameters for the different models. They have become frequency dependent since trainlength is frequency dependent.
        modelname = GxETrain
        model = Model(modelname)
        
        model.set_param_hint('sigma', value=s, vary=True)
        model.set_param_hint('mu', value=m, vary=True)
        model.set_param_hint('A',value=1.5, vary=True, min=0)
        model.set_param_hint('tau',value=200, vary=True, min=0)
        pars = model.make_params()
        
        modelname2 = GxESingleFold
        model2 = Model(modelname2)
        
        model2.set_param_hint('sigma', value=s, vary=True)
        model2.set_param_hint('mu', value=m, vary=True)
        model2.set_param_hint('A',value=1.5, vary=True, min=0)
        model2.set_param_hint('tau',value=200, vary=True, min=0)
        model2.set_param_hint('trainlength',value=trainlength, vary=False)
        pars2 = model2.make_params()
        
        
        modelname3 = EMG
        model3 = Model(modelname3)
        
        model3.set_param_hint('sigma', value=s, vary=True)
        model3.set_param_hint('mu', value=m, vary=True)
        model3.set_param_hint('A',value=1.5, vary=True, min=0)
        model3.set_param_hint('tau',value=200, vary=True, min=0)
        pars3 = model3.make_params()
     
     
        modelpow = Model(PLM)
        
        xsec = np.linspace(0,pulseperiod,P)
    
#        sp = 4 #number of subplots per figure 
#        numFig = k/4 + 1
#        totFig = int(len(taubins)/sp) + 1
#        print "Figure nr: %d/%d" % (numFig,totFig)
#        plt.figure(count*numruns + numFig, figsize=(14,9))
#        subplotcount = (k+1) - sp*(numFig-1)
#        plt.subplot(2,2,subplotcount)
#        plt.subplots_adjust(hspace=.3)             
#        plt.plot(xsec,observedlowpass[k]/np.max(observednonoise[k]),'y-')
    
    ## Fit Model 1:                    
        result = model.fit(observedlowpass[k],pars,x=xax)
#        print(result.fit_report(show_correl = False))
        besttau = result.best_values['tau']
        taustd = result.params['tau'].stderr
        obtainedtau.append(besttau)  
        obtainedtaustd.append(taustd)
        results.append(result)
               
#        plt.plot(xsec,result.best_fit/np.max(observednonoise[k]),'r-', linewidth = 2.0,label = r'%s $\tau=%.4f$' % ('Pre-fold', obtainedtau[k]*bs))     
    
#    ## Fit Model 2:    
#        result2 = model2.fit(observedlowpass[k],pars2,x=xax)
#        besttau2 = result2.best_values['tau']
#        taustd2 = result2.params['tau'].stderr              
#        obtainedtau2.append(besttau2)
#        obtainedtaustd2.append(taustd2)
#        results2.append(result2)
#        
##        plt.plot(xsec,result2.best_fit/np.max(observednonoise[k]),'b--',linewidth = 2.0, label = r'%s, $\tau=%.4f$' % ('Post-fold', obtainedtau2[k]*bs)) 
#    
#    ## Fit Model 3:     
#        result3 = model3.fit(observedlowpass[k],pars3,x=xax)
#        besttau3 = result3.best_values['tau']
#        taustd3 = result3.params['tau'].stderr                       
#        obtainedtau3.append(besttau3)
#        obtainedtaustd3.append(taustd3)
#        results3.append(result3)
##         
##        plt.plot(xsec,result3.best_fit/np.max(observednonoise[k]),'g--',linewidth = 2.0, label = r'%s, $\tau=%.4f$' % ('EMG' , obtainedtau3[k]*bs)) 
#
#    
#    ## Fit Check with Model 1:        
#        resultcheck = model.fit(checkfloors[k],pars,x=xax)  
##        print(resultcheck.fit_report(show_correl = False))                  
#        checktau = resultcheck.best_values['tau']
#        checktaustd = resultcheck.params['tau'].stderr
#        obtainedchecktau.append(checktau)
#        obtainedchecktaustd.append(checktaustd)
#        resultscheck.append(resultcheck)
#        
#        plt.plot(xsec,resultcheck.best_fit/np.max(observednonoise[k]),'c--',linewidth = 2.0, label = r'%s, $\tau=%.4f$' % ('Pre-fold check' , obtainedchecktau[k]*bs)) 

      
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
    obtainedtau2 = np.array(obtainedtau2) 
    obtainedtau3 = np.array(obtainedtau3)
    obtainedchecktau = np.array(obtainedchecktau)
       
    obtainedtaustd = np.array(obtainedtaustd)
    obtainedtaustd2 = np.array(obtainedtaustd2)
    obtainedtaustd3 = np.array(obtainedtaustd3)
    obtainedchecktaustd = np.array(obtainedchecktaustd)
    

    resultpow = modelpow.fit(obtainedtau,x=nurange,K=0.001,k=4)        
    specfit = resultpow.best_values['k']
    print 'alpha_1 = %.4f' %specfit
     
#    resultpow2 = modelpow.fit(obtainedtau2,x=nurange,K=0.001,k=4)
#    specfit2 = resultpow2.best_values['k']
#    print 'alpha_2 = %.4f' %specfit2    
#       
#    resultpow3 = modelpow.fit(obtainedtau3,x=nurange,K=0.001,k=4)
#    specfit3 = resultpow3.best_values['k']
#    print 'alpha_3 = %.4f' %specfit3            
#    
#    resultpowcheck = modelpow.fit(obtainedchecktau,x=nurange,K=0.001,k=4)
#    specfitcheck = resultpowcheck.best_values['k']
#    print 'alpha_ch = %.4f' %specfitcheck            
        
#    """Plots for different bandwidths"""    
#    taupow4 = psr.tau(Dval,Dsval,kappa1,nurange,light)
#    taupow4bins = taupow4*P/pulseperiod
#    
#    # Frequency range as determined by the chosen bandwidth
#    bwhigh = nurange+bandw/2.
#    bwlow = nurange-bandw/2.
#    
#    rfreq = 10**((np.log10(bwhigh)+ np.log10(bwlow))/2)
#    nurangeMHz = nurange*1000
#    
#    bin2sec = pulseperiod/P
#    bs= bin2sec
#
#    bwhigh2 = nurange+0.02/2.
#    bwlow2 = nurange-0.02/2.
#    rfreq2 = 10**((np.log10(bwhigh2)+ np.log10(bwlow2))/2)               
#    resultpowrf = modelpow.fit(obtainedtau,x=rfreq,K=0.001,k=4)
#    resultpowrf2 = modelpow.fit(obtainedtau,x=rfreq2,K=0.001,k=4)        
#    specfitrf = resultpowrf.best_values['k']
#    specfitrf2 = resultpowrf2.best_values['k']
#       
#    ticksMHz = []
#    for i in range(0,len(nurange),1):
#        tMHz = int(round(nurange[i]*1000))
#        ticksMHz.append(tMHz)
#    ticksMHz.append(int(round(1000*nurange[len(nurange)-1])))
#    ticksMHz = ticksMHz[0:len(ticksMHz):2]
#    
#    plt.figure((count+1)*100,figsize=(14, 9))
#    plt.errorbar(nurangeMHz,obtainedtau*bs,yerr=obtainedtaustd*bs,fmt='ro',markersize=8, capthick=2,linewidth=1.5, label = r'%s: $\alpha=$ %.3f' % (r'$f_m$',specfit))
#    plt.plot(nurangeMHz,resultpow.best_fit*bs,'r--',linewidth = 2.0)
#    
#    plt.errorbar(rfreq*1000,obtainedtau*bs,yerr=obtainedtaustd*bs,fmt='k^',markersize=10, capthick=2,linewidth=1.5, label = r'%s: $\alpha=$ %.3f, $\Delta f$= 50MHz' % (r'$f_c$' , specfitrf))        
#    plt.plot(rfreq*1000,resultpowrf.best_fit*bs,'k-',linewidth = 2.0)
#    
#    plt.errorbar(rfreq2*1000,obtainedtau*bs,yerr=obtainedtaustd*bs,fmt='c*',markersize=10, capthick=2,linewidth=1.5, label = r'%s: $\alpha=$ %.3f, $\Delta f$= 20MHz' % (r'$f_c$' , specfitrf2))
#    plt.plot(rfreq2*1000,resultpowrf2.best_fit*bs,'c-',linewidth = 2.0)    
#    
#    plt.xscale('log')
#    plt.yscale('log')
#    plt.xticks(ticksMHz,ticksMHz,fontsize=20)
#    plt.yticks(fontsize=20)
#    plt.xlabel('frequency (MHz)', fontsize=26)
#    plt.ylabel(r'$\tau$ (sec)',fontsize=26)
#    plt.legend(loc ='best',fontsize=22)
#    plt.xlim(40,120)
#    """ plots for different bandwidth end here"""  

#    ticksMHz = []
#    for i in range(0,len(nurange),1):
#        tMHz = int(round(nurange[i]*1000))
#        ticksMHz.append(tMHz)
#    ticksMHz.append(int(round(1000*nurange[len(nurange)-1])))
#    ticksMHz = ticksMHz[0:len(ticksMHz):2]
#    
#  
#    
#    plt.figure((count+1)*100,figsize=(14, 9))
#    plt.errorbar(nurange*1000,obtainedtau*bs,yerr=obtainedtaustd*bs,fmt='ro',markersize=8, capthick=2,linewidth=1.5, label = r'%s: $\alpha=$ %.3f' % ('Pre-fold' , specfit))
##    plt.errorbar(nurange*1000,obtainedtau2*bs,yerr=obtainedtaustd2*bs,fmt='b^',markersize=10,capthick=2, linewidth = 1.5,label = r'%s, $\alpha=$ %.3f' % ('Post-fold, long', specfit2))
#    plt.errorbar(nurange*1000,obtainedtau3*bs,yerr=obtainedtaustd3*bs,fmt='g*',markersize=8,capthick=2, linewidth = 1.5,label = r'%s, $\alpha=$ %.3f' % ('Folded Exponential', specfit3))
#    plt.errorbar(nurange*1000,obtainedtau4*bs,yerr=obtainedtaustd4*bs,fmt='m^',markersize=8,capthick=2, linewidth = 1.5,label = r'%s, $\alpha=$ %.3f' % ('Pre-fold,short', specfit4))
#    plt.errorbar(nurange*1000,obtainedchecktau*bs,yerr=obtainedchecktaustd*bs,fmt='co',markersize=8,capthick=2, linewidth = 1.5,label = r'%s, $\alpha=$ %.3f' % ('Check', specfitcheck))       
##    plt.errorbar(nurange*1000,obtainedchecktau2*bs,yerr=obtainedchecktaustd2*bs,fmt='ko',markersize=8,capthick=2, linewidth = 1.5,label = r'%s, $\alpha=$ %.3f' % ('Check 2', specfitcheck2))       
#    plt.plot(nurange*1000,resultpow.best_fit*bs,'r--',linewidth = 2.0)
##    plt.plot(nurange*1000,resultpow2.best_fit*bs,'b--',linewidth = 2.0)
#    plt.plot(nurange*1000,resultpow3.best_fit*bs,'g--',linewidth = 2.0)
#    plt.plot(nurange*1000,resultpow4.best_fit*bs,'m--',linewidth = 2.0)
#    plt.plot(nurange*1000,resultpowcheck.best_fit*bs,'c--',linewidth = 2.0)
##    plt.plot(nurange*1000,resultpowcheck2.best_fit*bs,'k--',linewidth = 2.0)
#     #for i in range(len(taubins)):        
#    #    plt.text(1000*nurange[i],1.1*bs*obtainedtau[i],'%.1f' % percarray[i])        
#    plt.plot(nurange*1000,taubins*bs,'y*-',markersize=8.0,linewidth=1.5,label = r'$\alpha= 4.00$')        
#    plt.xlim(nurange[0]*1000-5,nuhigh*1000+5)
##    plt.xscale('log')
#    #plt.yscale('log')
#    plt.xticks(ticksMHz,ticksMHz,fontsize=20)
#    plt.yticks(fontsize=20)
#    plt.xlabel('frequency (MHz)', fontsize=26)
#    plt.ylabel(r'$\tau$ (sec)',fontsize=26)
#    #plt.legend(loc ='lower left',fontsize=18)
#    plt.legend(loc ='best',fontsize=18)
    

#    plt.figure((count+1)*1000,figsize=(14, 9))
#    plt.errorbar(nurange*1000,obtainedtau*bs,yerr=obtainedtaustd*bs,fmt='ro',markersize=8, capthick=2,linewidth=1.5, label = r'%s: $\alpha=$ %.3f' % ('Pre-fold' , specfit))
#    plt.errorbar(nurange*1000,obtainedtau2*bs,yerr=obtainedtaustd2*bs,fmt='b^',markersize=10,capthick=2, linewidth = 1.5,label = r'%s, $\alpha=$ %.3f' % ('Post-fold, long', specfit2))
##    plt.errorbar(nurange*1000,obtainedtau3*bs,yerr=obtainedtaustd3*bs,fmt='g*',markersize=8,capthick=2, linewidth = 1.5,label = r'%s, $\alpha=$ %.3f' % ('EMG', specfit3))
#    plt.errorbar(nurange*1000,obtainedchecktau*bs,yerr=obtainedchecktaustd*bs,fmt='co',markersize=8,capthick=2, linewidth = 1.5,label = r'%s, $\alpha=$ %.3f' % ('Pre-fold check', specfitcheck))
#    plt.plot(nurange*1000,resultpow.best_fit*bs,'r--',linewidth = 2.0)
#    plt.plot(nurange*1000,resultpow2.best_fit*bs,'b--',linewidth = 2.0)
##    plt.plot(nurange*1000,resultpow3.best_fit*bs,'g--',linewidth = 2.0)
#    plt.plot(nurange*1000,resultpowcheck.best_fit*bs,'c--',linewidth = 2.0)  
#    plt.plot(nurange*1000,taubins*bs,'y*-',markersize=8.0,linewidth=1.5,label = r'$\alpha= 4.00$')    
#    plt.xlim(nurange[0]*1000-5,nuhigh*1000+5)
#    plt.xscale('log')
#    plt.yscale('log')
#    plt.xticks(ticksMHz,ticksMHz,fontsize=20)
#    plt.yticks(fontsize=20)
#    plt.xlabel('frequency (MHz)', fontsize=26)
#    plt.ylabel(r'$\tau$ (sec)',fontsize=26)
#    #plt.legend(loc ='lower left',fontsize=18)
#    plt.legend(loc ='best',fontsize=18)

#     
#    alpha1.append(specfit)
##    alpha2.append(specfit2) 
##    alpha3.append(specfit3)
##    alphacheck.append(specfitcheck)
#    
#    resultpows.append(resultpow)
##    resultpows2.append(resultpow2)
##    resultpows3.append(resultpow3)
##    resultpowscheck.append(resultpowcheck)
##    
#    observedcombo.append(observed)
#    resultcombo.append(results)
#    result2combo.append(results2)
#    result3combo.append(results3)
#    resultcheckcombo.append(resultscheck)
#    observedlowpasscombo.append(observedlowpass)
#  
#    obtainedtaulist.append(obtainedtau)
#    obtainedtaustdlist.append(obtainedtaustd)
#    obtainedtau2list.append(obtainedtau2)
#    obtainedtau2stdlist.append(obtainedtaustd2)
#    obtainedtau3list.append(obtainedtau3)
#    obtainedtau3stdlist.append(obtainedtaustd3)
#    obtainedtauchecklist.append(obtainedchecktau)
#    obtainedtaucheckstdlist.append(obtainedchecktaustd)
#
#
#print np.mean(alpha1)
#print np.std(alpha1)
#
#print np.mean(alpha2)
#print np.std(alpha2)
#
#print np.mean(alpha3)
#print np.std(alpha3)
#
#print np.mean(alphacheck)
#print np.std(alphacheck)
#
#print 'snr: '+str(snr)
#print 'pulseperiod: '+str(pulseperiod)
#print 'dutycycle: '+str(dutycycle)
#print 'numruns: '+str(numruns)
#print 'nurange:'+str(nulow)+'-'+str(nuhigh)+' GHz'+'('+str(incr)+')'
##print 'edge:'+str(edge)    
#print"****************************************"
#print"****************************************"
# 
#
##np.savetxt('Pythondata/taulists/FastPulsar/alpha1_%s.txt' % timestr, alpha1)
##np.savetxt('Pythondata/taulists/FastPulsar/alpha2_%s.txt' % timestr, alpha2) 
##np.savetxt('Pythondata/taulists/FastPulsar/alpha3_%s.txt' % timestr, alpha3) 
##np.savetxt('Pythondata/taulists/FastPulsar/alphacheck_%s.txt' % timestr, alphacheck) 
#
#
#   
### Find the numrun iteration that represents the mean of all the runs best
#    
#res1 = np.abs(np.mean(alpha1)-alpha1)
#res2 =  np.abs(np.mean(alpha2)-alpha2)
#res3 = np.abs(np.mean(alpha3)-alpha3)
#resch= np.abs(np.mean(alphacheck)-alphacheck)
#
#
### Choose the set of alpha's with least overall difference from mean
### I.e the set that represents the mean best
#
#ind_old = np.argmin(res1+resch)
##
#### Doen hierdie check vir interessantheid:
##
##in1 = np.argmin(res1)
##in2 = np.argmin(res2)
##in3 = np.argmin(res3)
##inch = np.argmin(resch)
##
##print 'Are all these indexes the same?'
##print ind,in1,in2,in3,inch
#
#"""Change this into some weighted mean for improved accuracy"""
#
#weightval = np.linspace((1./numruns),1,numruns)
#weights = weightval/np.sum(weightval)
#
#indd = np.arange(0,numruns,1)
#
#stack1 = np.column_stack((indd,res1))
#stackch = np.column_stack((indd,resch))
#
#sort1 = stack1[stack1[:,1].argsort()]
#sort1_w = np.column_stack((sort1[:,0],weights[::-1])) ##::-1 just gives the reverese order of the weights, as I want the index associated with the smallest residual to have the largest weight
#sort1_ind = sort1_w[sort1_w[:,0].argsort()]
##sort1[:,1] *= weights
#
#sortch = stackch[stackch[:,1].argsort()]
#sortch_w = np.column_stack((sortch[:,0],weights[::-1]))
#sortch_ind = sortch_w[sortch_w[:,0].argsort()]
#
#ind = np.argmax(sortch_ind[:,1]*sort1_ind[:,1])
#
#
### Plot tau spectrum
#
#
#ticksMHz = []
#for i in range(0,len(nurange),1):
#    tMHz = int(round(nurange[i]*1000))
#    ticksMHz.append(tMHz)
#ticksMHz.append(int(round(1000*nurange[len(nurange)-1])))
#ticksMHz = ticksMHz[0:len(ticksMHz):2]
#
#
#plt.figure((count+1),figsize=(14, 9))
#plt.errorbar(nurange*1000,obtainedtaulist[ind]*bs,yerr=obtainedtaustdlist[ind]*bs,fmt='ro',markersize=8, capthick=2,linewidth=1.5, label = r'%s: $\alpha=$ %.3f' % ('Pre-fold' , alpha1[ind]))
#plt.errorbar(nurange*1000,obtainedtau2list[ind]*bs,yerr=obtainedtau2stdlist[ind]*bs,fmt='b^',markersize=10,capthick=2, linewidth = 1.5,label = r'%s, $\alpha=$ %.3f' % ('Post-fold, long', alpha2[ind]))
#plt.errorbar(nurange*1000,obtainedtau3list[ind]*bs,yerr=obtainedtau3stdlist[ind]*bs,fmt='g*',markersize=8,capthick=2, linewidth = 1.5,label = r'%s, $\alpha=$ %.3f' % ('EMG', alpha3[ind]))
#plt.errorbar(nurange*1000,obtainedtauchecklist[ind]*bs,yerr=obtainedtaucheckstdlist[ind]*bs,fmt='co',markersize=8,capthick=2, linewidth = 1.5,label = r'%s, $\alpha=$ %.3f' % ('Pre-fold check', alphacheck[ind]))
#plt.plot(nurange*1000,resultpows[ind].best_fit*bs,'r--',linewidth = 2.0)
#plt.plot(nurange*1000,resultpows2[ind].best_fit*bs,'b--',linewidth = 2.0)
#plt.plot(nurange*1000,resultpows3[ind].best_fit*bs,'g--',linewidth = 2.0)
#plt.plot(nurange*1000,resultpowscheck[ind].best_fit*bs,'c--',linewidth = 2.0)  
#plt.plot(nurange*1000,taubins*bs,'y*-',markersize=8.0,linewidth=1.5,label = r'$\alpha= 4.00$')    
#plt.xlim(nurange[0]*1000-5,nuhigh*1000+5)
#plt.xscale('log')
#plt.yscale('log')
#plt.xticks(ticksMHz,ticksMHz,fontsize=20)
#plt.yticks(fontsize=20)
#plt.xlabel('frequency (MHz)', fontsize=26)
#plt.ylabel(r'$\tau$ (sec)',fontsize=26)
##plt.legend(loc ='lower left',fontsize=18)
#plt.legend(loc ='best',fontsize=22)
#
##filename = 'NewTestTauspectrum'+str(nulow)+'_'+str(nuhigh)+'.png'
##picpath = "/Users/marisa/Documents/PhD/GitHub_Scattering/Plots/FastPulsar"
##fileoutput = os.path.join(picpath,filename)
##plt.savefig(fileoutput, dpi=150)
#
#
### Arrays to be saved:
#tau1 = obtainedtaulist[ind]
#tau2 = obtainedtau2list[ind]
#tau3 = obtainedtau3list[ind]
#tauch = obtainedtauchecklist[ind]
#
#tau1std = obtainedtaustdlist[ind]
#tau2std = obtainedtau2stdlist[ind]
#tau3std = obtainedtau3stdlist[ind]
#tauchstd = obtainedtaucheckstdlist[ind]
#
##np.savetxt('Pythondata/taulists/FastPulsar/tau1_%s.txt' %timestr, tau1)
##np.savetxt('Pythondata/taulists/FastPulsar/tau2_%s.txt' %timestr, tau2) 
##np.savetxt('Pythondata/taulists/FastPulsar/tau3_%s.txt' %timestr, tau3) 
##np.savetxt('Pythondata/taulists/FastPulsar/tauch_%s.txt' %timestr, tauch) 
##
##np.savetxt('Pythondata/taulists/FastPulsar/tau1std_%s.txt' %timestr, tau1std) 
##np.savetxt('Pythondata/taulists/FastPulsar/tau2std_%s.txt' %timestr, tau2std) 
##np.savetxt('Pythondata/taulists/FastPulsar/tau3std_%s.txt' %timestr, tau3std) 
##np.savetxt('Pythondata/taulists/FastPulsar/tauchstd_%s.txt' %timestr, tauchstd)  
#  
#
### Plot profiles
#sp = 4 #number of subplots per figure 
#
#for k in range(0,len(taubins)):
#    numFig = k/4 + 1
#    totFig = int(len(taubins)/sp) + 1
#    print "Figure nr: %d/%d" % (numFig,totFig)
#    plt.figure(count*numruns + numFig, figsize=(17,10))
#    subplotcount = (k+1) - sp*(numFig-1)
#    plt.subplot(2,2,subplotcount)
#    plt.subplots_adjust(hspace=.4)             
#    plt.plot(xsec,observedlowpasscombo[ind][k]/np.max(observednonoise[k]),'y-')
#    plt.plot(xsec,resultcombo[ind][k].best_fit/np.max(observednonoise[k]),'r-', linewidth = 2.0,label = r'%s $\tau=%.4f \pm %.4f$' % ('Pre-fold', obtainedtaulist[ind][k]*bs,obtainedtaustdlist[ind][k]*bs))     
#    plt.plot(xsec,result2combo[ind][k].best_fit/np.max(observednonoise[k]),'b--',linewidth = 2.0, label = r'%s, $\tau=%.4f \pm %.4f$' % ('Post-fold', obtainedtau2list[ind][k]*bs,obtainedtau2stdlist[ind][k]*bs)) 
#    plt.plot(xsec,result3combo[ind][k].best_fit/np.max(observednonoise[k]),'g--',linewidth = 2.0, label = r'%s, $\tau=%.4f \pm %.4f$' % ('EMG', obtainedtau3list[ind][k]*bs,obtainedtau3stdlist[ind][k]*bs)) 
#    plt.plot(xsec,resultcheckcombo[ind][k].best_fit/np.max(observednonoise[k]),'c--',linewidth = 2.0, label = r'%s, $\tau=%.4f \pm %.4f$' % ('Pre-fold check' ,obtainedtauchecklist[ind][k]*bs,obtainedtaucheckstdlist[ind][k]*bs)) 
#    plt.title(r'%.0f MHz, simulated $\tau=%.4f$, ' % (1000*nurange[k],tauval[k]),fontsize=20) 
#    plt.legend(loc = 'upper right',fontsize=13)
#    plt.xlabel('time (sec)',fontsize=20)
#    plt.xticks(fontsize=18)
#    plt.yticks(fontsize=18)
#    plt.xlim(0,pulseperiod)
#
##    filename = 'NewTestProfiles'+str(numFig)+'_HF.png'
##    picpath = "/Users/marisa/Documents/PhD/GitHub_Scattering/Plots/FastPulsar"
##    fileoutput = os.path.join(picpath,filename)
##    plt.savefig(fileoutput, dpi=150)   
#
### Arrays to be saved:
#    arNoN = observednonoise[k]
#    arN = observedlowpasscombo[ind][k]
#    ar1 = resultcombo[ind][k].best_fit
#    ar2 = result2combo[ind][k].best_fit
#    ar3 = result3combo[ind][k].best_fit
#    arch = resultcheckcombo[ind][k].best_fit 
#
##    np.savetxt('Pythondata/arNoN/FastPulsar/arNoN_%d_%s.txt' %(k,timestr), arNoN)
##    np.savetxt('Pythondata/arN/FastPulsar/arN_%d_%s.txt' %(k,timestr), arN)
##    np.savetxt('Pythondata/ar1/FastPulsar/ar1_%d_%s.txt' %(k,timestr), ar1)
##    np.savetxt('Pythondata/ar2/FastPulsar/ar2_%d_%s.txt' %(k,timestr), ar2)
##    np.savetxt('Pythondata/ar3/FastPulsar/ar3_%d_%s.txt' %(k,timestr), ar3)
##    np.savetxt('Pythondata/arch/FastPulsar/arch_%d_%s.txt' %(k,timestr), arch)
#    
##arsec = xsec
##np.savetxt('Pythondata/arsec/arsec_B.txt', arsec)
#
##print resminarray        
#
#
##        percmin = np.abs(resminarray/np.min(climb)*100)
##        perctaures = np.abs(np.divide((obtainedchecktau - obtainedtau),obtainedchecktau))*100
#
##        combinedstd = np.sqrt(np.power(obtainedtaustd,2) + np.power(obtainedchecktaustd,2))
##        errorbar = np.divide(combinedstd,obtainedchecktau)*100
#
#
##plt.figure(figsize=(14, 9))
##plt.errorbar(nurange*1000,obtainedtau*bs,yerr=obtainedtaustd*bs,fmt='ro',markersize=10, capthick=2,linewidth=1.5, label = r'%s: $\alpha=$ %.3f' % (r'$f_m$' , specfit))
##plt.errorbar(nurange*1000,obtainedtau2*bs,yerr=obtainedtaustd2*bs,fmt='b^',markersize=10,capthick=2, linewidth = 1.5,label = r'%s, $\alpha=$ %.3f' % ('Post-fold, long', specfit2))
##plt.errorbar(nurange*1000,obtainedtau3*bs,yerr=obtainedtaustd3*bs,fmt='g*',markersize=10,capthick=2, linewidth = 1.5,label = r'%s, $\alpha=$ %.3f' % ('Folded Exponential', specfit3))
##plt.errorbar(nurange*1000,obtainedchecktau*bs,yerr=obtainedchecktaustd*bs,fmt='co',markersize=10,capthick=2, linewidth = 1.5,label = r'%s, $\alpha=$ %.3f' % ('Check', specfitcheck))       
##plt.plot(nurange*1000,resultpow.best_fit*bs,'r--',linewidth = 2.0)
##plt.plot(nurange*1000,resultpow2.best_fit*bs,'b--',linewidth = 2.0)
##plt.plot(nurange*1000,resultpow3.best_fit*bs,'g--',linewidth = 2.0)
##plt.plot(nurange*1000,resultpowcheck.best_fit*bs,'c--',linewidth = 2.0)  
##plt.plot(nurange*1000,taubins*bs,'m*-',markersize=8.0,linewidth=1.5,label = r'$\alpha= 4.00$')        
##plt.xlim(nulow*1000,nuhigh*1000)
##plt.xscale('log')
##plt.yscale('log')
##plt.xticks(ticksMHz,ticksMHz,fontsize=20)
##plt.yticks(fontsize=20)
##plt.xlabel('frequency (MHz)', fontsize=26)
##plt.ylabel(r'$\tau$ (sec)',fontsize=26)
##plt.legend(loc ='lower left',fontsize=18)
