#!/usr/bin/python
"""@author: Marisa Geyer"""

"""This script analyses the impact of using a low pass filter to estimate the minimum of a noisy scattered profile."""
"""It compares the model fitted tau value when the minimum is known ai priori to the tau value obtained 
when the minimum is estimated using a low pass filter"""


"""Import required modules"""
import pypsr_redo as psr
import scatmodels_nolf as sm
import os, sys
import matplotlib.pyplot as plt
from lmfit import Model, conf_interval, printfuncs
from lmfit import minimize, Parameter, Parameters, fit_report
import numpy as np
import scipy.ndimage.filters as snf

"""Choose a fitting model"""
#### MODEL 1 #####################
def GxETrain(x,mu,sigma, A, tau):
    #This model convolves a pulsetrain with a broadening function
    #It extracts one of the last convolved profiles, subtracts the climbed baseline and then adds noise to it
    bins, profile = psr.makeprofile(nbins = P, ncomps = 1, amps = A, means = mu, sigmas = sigma)
    binstau = np.linspace(1,P,P)
    scat = psr.psrscatter(psr.broadfunc(binstau,tau),psr.pulsetrain(3, bins, profile))   
    climb, observed_nonoise, rec, flux = psr.extractpulse(scat, 2, P)
    return observed_nonoise

"""Moving average is no longer used"""
"""def movingaverage(interval, window_size):
    window= np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')"""


""" Set constants"""
light = 9.71561189*10**-12      #Speed of light in kpc/s  
mrad = 4.85*10**-9              #Convertion from miliarcseconds to radians

""" Define global parameters"""
global nbinspow  
global P
#global lf

"""Resolution is chosen to be 512 throughout"""
nbinspow = 9
P = 2**nbinspow   #### note that this is independently put into the functions. Make sure they match!
print 'bins: '+str(P)


""" Input observed frequency or frequency range"""
incr = 0.03
nulow, nuhigh = float(0.06),float(0.10)
nurange = np.arange(nulow,nuhigh,incr)
print 'nurange: %.2f - %.2f GHz' % (nurange[0], nurange[-1])


"""
###############################################################################
## Create the properties of the ingoing pulse
###############################################################################
"""


pulseperiod =1.0                  #in seconds
dutycycle = float(5.0)             #as a % of the overall pulseperiod
m = float(P/4)                      #Let the mean of the pulse be at a 1/4 overall bins
w50 = float((dutycycle/100)*P)      #FWHM
s = w50/(2*np.sqrt(2*np.log(2)))    #sigma calculated through the dutycycle
a = 1                               #amplitude. at some point this will be replaced by the intrinsic spectrum. 

snr = 20
print 'snr: '+str(snr)

bin2sec = pulseperiod/P
bs= bin2sec

"""Create intrinsic profile with spectral index 1.6. Normalise to have flux = 1.0 at lowest freq."""
## This makes use of functions from pypsr_redo.py

bins, profile = psr.makeprofile(nbins = P, ncomps = 1, amps = a, means = m, sigmas = s)

spectralindex = 1.6  #Input spectral index as a postivie number, for nu^-alpha
profile_intr = psr.profilespec(nurange,spectralindex,profile)
profile_intr_norm = profile_intr/np.sum(profile_intr[0])   #Normalised such that the max intrinsic flux = 1.0. That is the intrinsice pulse at the lowest frequency has flux = 1

"""
###############################################################################
## SCATTERING SETUP
###############################################################################
"""
#Dval is the distance from the pulsar to the observer in kpc
#Dsval is the distance from the pulsar to the scattering screen in kpc
# k1 sets the scattering strength in the cold plasma proportionality: sigma propto freq^-2 in mas
# kappa1 converts k1 to mrad for calculations
# tauval gives the tau in seconds at a given freq
# taubins converts tauval from seconds to bins

Dval, Dsval = float(3.0),float(1.5)
k1 = 3.0
kappa1 = k1*mrad

tauval = psr.tau(Dval,Dsval,kappa1,nurange,light)
taubins = (tauval/pulseperiod)*P

"""
###############################################################################
## CONVOLVE INGOING PULSE AND BROADENING FUNC. AND EXTRACT STABILISED PROFILE
###############################################################################
"""
##This uses the broadening function broadfunc from pypsr_redo.py
##as well as psrscatter from pypsr which does the convolution
##Gaussian noise with a chosen snr is added to the scattered signal

xax = np.arange(1,P,1)

for i in range(0,len(taubins)):
    scat = psr.psrscatter(psr.broadfunc(xax,taubins[i]),psr.pulsetrain(3, bins, profile_intr_norm[i])) 
    climb, observed_nonoise, rec, flux = psr.extractpulse(scat, 2, P)
    peak = np.max(observed_nonoise)
    
#    plt.figure()
#    plt.plot(scat)
#    plt.title('Has the climb stabilised?, Obs freq: %.0f MHz' % (1000*nurange[i]))
    
    resmedminarray = []
    reslpminarray = []
    
    filtertaus = []
    checktaus = []
    lowpasstaus = []
    filtertaustdarray = []
    checktaustdarray = []
    lowpasstaustdarray = []    
    
    totruns = 30;
    for count in range(totruns):
        print '%d/%d' %(count,totruns)
        noise = np.random.normal(0,peak/snr,P)
        observedadd = climb + noise
        idmin = np.argmin(observedadd)
  
        """Method 1: Median Filter"""        
#        filterwindow = 300
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
#        plt.figure()
#        plt.plot(medfilter - filmin,'g--',linewidth=2.0,label = 'Moving Median Filter')
#        plt.plot(observed_nonoise[fstart:fend],'r',linewidth=2.0,label = 'No noise data')
#        plt.ylim(-0.0002)
#        plt.legend()

#        if abs(resmedmin) > 1e-05:
#            print ("Residual between minima exceeds 1e-5.")      


        """Method 2: low Pass Filter"""
#        edge = 15  ##works better for 90MHz (higher freq)
#        edge = 20 #works better for 60MHz (lower freq)
        edge = 20 -5*i
        fourier = np.fft.fft(observedadd)
        fouriersubset = np.concatenate((fourier[0:edge],fourier[P-edge:P]))
        invfourier = np.fft.ifft(fouriersubset)
        binfix = np.linspace(1,P,len(invfourier))
        scalefix = len(invfourier)/float(P)
        lowpasssignal = np.real(scalefix*invfourier)
        lowpassmin = np.min(lowpasssignal)
        observedzerolowpass = observedadd - lowpassmin
        reslpmin = np.min(climb) - lowpassmin
        perclpmin = np.abs(reslpmin/np.min(climb)*100)
        reslpminarray.append(reslpmin)

        observedcheck = observed_nonoise +noise
        
#        plt.figure()
#        plt.plot(climb,'r')
#        plt.plot(medfilter,'b')
#        
#        plt.figure()
#        plt.plot(binfix,lowpasssignal - lowpassmin, 'm--', linewidth=2.0,label = 'Low Pass Filter')        
#        plt.plot(observed_nonoise,'r',linewidth=2.0,label = 'No noise data')
#        plt.ylim(-0.0002)
#        plt.legend()      
       
        #sys.exit()
        ###############################################################################
        ## FIT BROADENED PROFILE TO EXTRACT TAU VALUE
        ###############################################################################
        
        # Initialise the parameter values of the different models
         
        modelname = GxETrain
        model = Model(modelname)
        
        model.set_param_hint('sigma', value=s, vary=True)
        model.set_param_hint('mu', value=m, vary=True)
        model.set_param_hint('A',value=1.5, vary=True, min=0)
        model.set_param_hint('tau',value=200, vary=True, min=0)
        pars = model.make_params()
        
        xsec = np.linspace(0,pulseperiod,P)
        
        ## FIT
        
#        resultmed = model.fit(observedzeromed,pars,x=xax)
        resultlowpass = model.fit(observedzerolowpass,pars,x=xax)
        resultcheck = model.fit(observedcheck,pars,x=xax)
#        print(result.fit_report(show_correl = False))
#        print(resultcheck.fit_report(show_correl = False))
        
#        filtertau = resultmed.best_values['tau']
        lowpasstau = resultlowpass.best_values['tau']
        checktau = resultcheck.best_values['tau']
        
#        filtertaustd = resultmed.params['tau'].stderr
        lowpasstaustd = resultlowpass.params['tau'].stderr
        checktaustd = resultcheck.params['tau'].stderr 
        
#        plt.figure()
#        plt.plot(xax,observedzero,'y')
#        plt.plot(xax,result.best_fit,'b--',linewidth = 2.0, label = r'%s, $\tau=%.4f$' % (modelname.__name__ , filtertau)) 
#        plt.legend()
#        plt.title('Sim. tau: %.2f, done with Median Filter' % taubins[0])
#        
#        plt.figure()
#        plt.plot(xax,observedcheck,'r')
#        plt.plot(xax,resultcheck.best_fit,'g--',linewidth = 2.0, label = r'%s, $\tau=%.4f$' % (modelname.__name__ , resultcheck.best_values['tau'])) 
#        plt.legend()
#        plt.title('Sim. tau: %.2f' % taubins[0])
#               
#        filtertaus.append(filtertau)
        lowpasstaus.append(lowpasstau)
        checktaus.append(checktau)
        
#        filtertaustdarray.append(filtertaustd)
        lowpasstaustdarray.append(lowpasstaustd)
        checktaustdarray.append(checktaustd)
        
    percmedminarray = np.abs(resmedminarray/np.min(climb)*100)
    perclpminarray = np.abs(reslpminarray/np.min(climb)*100)
#    percmedtaures = np.abs(np.divide((np.array(checktaus) - np.array(filtertaus)),checktaus))*100
    perclptaures = np.abs(np.divide((np.array(checktaus) - np.array(lowpasstaus)),checktaus))*100
    
#    medcombinedstd = np.sqrt(np.power(filtertaustdarray,2) + np.power(checktaustdarray,2))
    lpcombinedstd = np.sqrt(np.power(lowpasstaustdarray,2) + np.power(checktaustdarray,2))    
    
#    mederrorbar = np.divide(medcombinedstd,checktaus)*100
    lperrorbar = np.divide(lpcombinedstd,checktaus)*100
    
#    print 'Median filter'
#    print resmedmin, nurange[i], np.min(climb)
#    print percmedmin
#    print '*************'   
    
    print 'Low pass filter'
    print reslpmin, nurange[i], np.min(climb)
    print perclpmin
    print '*************' 
    
    plt.figure(figsize = (8,6.3))
#    plt.errorbar(percmedminarray,percmedtaures,yerr=mederrorbar, fmt ='bo', markersize=8,capthick=2,linewidth=1.5, label = 'Median Filter')
    plt.errorbar(perclpminarray,perclptaures,yerr=lperrorbar, fmt ='bo', markersize=8,capthick=2,linewidth=1.5)
#    plt.legend(fontsize=12, loc = 'best')    
    plt.xlabel('% error in determining the minimum',fontsize=18)
    plt.ylabel(r'% error in $\tau$ fit',fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.title(r'$f_m$: %.0f MHz' % (1000*nurange[i]), fontsize=20)
    filenameA = 'F%d_lowpass9.png' %(1000*nurange[i])
    picpath = "/Users/marisa/Dropbox/Aris/TexOutputs/ScatteringPaperTexAndImages/LowPassFilter"
    fileoutput = os.path.join(picpath,filenameA)
    plt.savefig(fileoutput, dpi=150)   

    
    binstot = np.array([50,16]);
    binsize = np.array([50*0.8/16.,0.8])    
    
#    plt.figure(figsize = (8,6.3))   
#    plt.hist(perclptaures,bins = np.arange(0,binstot[i],binsize[i]))
#    plt.xlabel(r'% error in determining $\tau$',fontsize=18)
#    plt.ylabel('count',fontsize=18)
#    plt.xticks(fontsize=16)
#    plt.yticks(fontsize=16)
#    filenameB = 'F%d_lowpass_histogram9.png' %(1000*nurange[i])
#    picpath = "/Users/marisa/Dropbox/Aris/TexOutputs/ScatteringPaperTexAndImages/LowPassFilter"
#    fileoutput = os.path.join(picpath,filenameB)
#    plt.savefig(fileoutput, dpi=150)   




#plt.figure(90)
#plt.plot(reslpminarray,label = 'Low Pass', linewidth = 2.0)
##plt.plot(percmedminarray,label = 'Median Filter', linewidth = 2.0)
#plt.xlabel('counts')
#plt.ylabel('error')
#plt.legend(loc = 'best')
#
#
#
#plt.figure(100)
#plt.plot(perclpminarray,label = 'Low Pass %.0f' % edge, linewidth = 2.0)
##plt.plot(percmedminarray,label = 'Median Filter', linewidth = 2.0)
#plt.xlabel('counts')
#plt.ylabel('% error')
#plt.legend(loc = 'best')
       
    
#plt.figure()
#plt.plot(xax,observedzero,'y')
#plt.plot(xax,result.best_fit,'b--',linewidth = 2.0, label = r'%s, $\tau=%.4f$' % (modelname.__name__ , filtertau/P*pulseperiod)) 
#plt.legend()
#plt.title('Sim. tau: %.4f, done with Median Filter' % tauval[-1])
#       
#plt.figure()
#plt.plot(xax,observedcheck,'r')
#plt.plot(xax,resultcheck.best_fit,'g--',linewidth = 2.0, label = r'%s, $\tau=%.4f$' % (modelname.__name__ , (resultcheck.best_values['tau'])/P*pulseperiod)) 
#plt.legend()
#plt.title('Sim. tau: %.4f' % tauval[-1])
     

#plt.figure()
#plt.errorbar(percmin,perctaures,yerr=errorbar, fmt ='bo', markersize=8,capthick=2,linewidth=1.5)
#plt.xlabel('% error in determining the minimum',fontsize=16)
#plt.ylabel(r'% error in $\tau$ fit',fontsize=16)
#plt.title(r'Median sampling rate of: P/%.0f; $f_m$: %.0f MHz, $f_c$: %.0f MHz' % (samp, 1000*nurange[i], 1000*rfreq[i]), fontsize=16)
