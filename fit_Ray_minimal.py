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
import time
#from photonscatrelinf import timeres, photonsinterfreq,photonsfreq, DD, Dss, sigma1,startfreq,endfreq, f_incr
from Ray_minimal import  binstimeres, probfreq, fluxtr, DD, Dss, sigma1,startfreq,endfreq, f_incr
#photonsinterpp, binstimeresinterp
## If want to fit at different resolution that Ray_minimal evaluated at, have to import timeressamp and probf.


#### MODEL 1 #####################
def GxETrain(x,mu,sigma, A, tau):
    trainlength = 20
    #This model convolves a pulsetrain with a broadening function
    #It extracts one of the last convolved profiles, subtracts the climbed baseline and then adds noise to it
    bins, profile = psr.makeprofile(nbins = P, ncomps = 1, amps = A, means = mu, sigmas = sigma)
    binstau = np.linspace(1,trainlength*P,trainlength*P)
    scat = psr.psrscatter(psr.broadfunc(binstau,tau),psr.pulsetrain(trainlength, bins, profile))   
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


def EMG(x,mu,sigma,A,tau):
    return (((1/tau)/2)*A*np.exp(((1/tau)/2)*(2*mu+(1/tau)*sigma**2-2*x))*special.erfc((mu+(1/tau)*sigma**2-x)/(np.sqrt(2)*sigma)))
    
def PLM(x,K,k):
    return K*pow(x,-k)


## Set up timer
starttime = time.time()

#photonf = []
#for i in range(len(photonsfreq)):
#    phrow = photonsfreq[i]
#    phsamp = phrow[0:len(phrow):4]
#    photonf.append(phsamp)
#
#photonf = np.array(photonf)

### Universal constants 
light = 9.71561189*10**-12      #Speed of light in kpc/s  
mrad = 4.85*10**-9              #Convertion from miliarcseconds to radians


### Define global parameters
global nbinspow  
global P
global lf

#nbinspow = 9
#P = 2**nbinspow   #### note that this is independently put into the functions. Make sure they match!
#print 'bins: '+str(P)

### Input the observed frequency range

bandw = float(5)/float(1000) #GHz
#cfreq = float(150)/float(1000) # center freq GHz

incr = f_incr
nulow, nuhigh = float(startfreq),float(endfreq)
nurange = np.arange(nulow,nuhigh+incr,incr)

print 'bandwidth: '+str(bandw)+' GHz'
print str(nulow)+'-'+str(nuhigh)+' GHz'



### Create the properties of the ingoing pulse

pulseperiod =1.0                  #in seconds
dutycycle = float(2.5)             #as a % of the overall pulseperiod

## Create the profile to have the same time resolution than the broadening function created by photonscat.py

P = int(pulseperiod/binstimeres)


m = float(P/4)                      #Let the mean of the pulse be at a 1/4 overall bins
w50 = float((dutycycle/100)*P)      #FWHM
s = w50/(2*np.sqrt(2*np.log(2)))    #sigma calculated through the dutycycle
a = 1                               #amplitude. at some point this will be replaced by the intrinsic spectrum. 

trainlength = 20

## Intrinsic profile




bins, profile = psr.makeprofile(nbins = P, ncomps = 1, amps = a, means = m, sigmas = s)
xaxlong =  np.linspace(1,20*P,20*P)
#xaxlong =  np.linspace(1,100*P,100*P)

spectralindex = 1.6  #Input spectral index as a postivie number, for nu^-alpha
profile_intr = psr.profilespec(nurange,spectralindex,profile)
profile_intr_norm = profile_intr/np.sum(profile_intr[0])   #Normalised such that the max intrinsic flux = 1.0. That is the intrinsice pulse at the lowest frequency has flux = 1

###############################################################################
## ISOTROPIC
###############################################################################

## In the case of importing data/broadening functions these parameters will only play a roll in plotting the alpha = 4.0 spectrum to compare
## In the case where I simulate the data to fit here, these paramteres will dicatate the shape of the broadening function

Dval, Dsval = float(DD),float(Dss)

k1 = sigma1
kappa1 = k1*mrad

tauval = psr.tau(Dval,Dsval,kappa1,nurange,light)


print tauval

#if tauval[0] > 10*pulseperiod:
#    sys.exit("tau warning: scattering is too high")
#else:
#    taubins = (tauval/pulseperiod)*P
taubins = (tauval/pulseperiod)*P
#
#print taubins
#print np.median(taubins)

#sys.exit()


##Create observed pulses by scattering a pulsetrain with an exponential broadening function
  
observed = []
observednonoise = []
observedflux = []
observedfluxfixed = []
scatt = []
#scintt = []

results = []

xsec = np.linspace(0,pulseperiod,P)



#np.savetxt('Pythondata/photonsfreq.txt',photonsfreq)

for i in range(len(nurange)):
    scat = psr.psrscatter(probfreq[i],psr.pulsetrain(3, bins, profile_intr_norm[i]))
    scatt.append(scat)      
#    climb, observed_nonoise, rec, flux = psr.extractpulse(scat, 2, P)
    climb, observed_nonoise, rec, flux = psr.extractpulse(scat, 0, P)
    observedflux.append(flux)
    observednonoise.append(observed_nonoise)

#sys.exit()
# the flux as produced by psr.extractpulse is normalised such that the lowest frequency has flux 1
# Profiles:
plt.figure()
for i in range(len(nurange)):    
    observedfluxfix = observednonoise[i]*fluxtr[i]
#    plt.figure()    
##    plt.plot(xsec,observedfluxfix/np.max(observedfluxfix),'b',linewidth = 4.0)
##    plt.plot(scatt[i]/np.max(scatt[i]),'m',linewidth = 4.0)
#    plt.plot(observednonoise[i]/np.max(observednonoise[i]),'g',linewidth = 4.0)
#    plt.xlabel("time(sec)",fontsize=16)
#    plt.ylabel("normalized intensity",fontsize=16)
#    plt.xticks(fontsize=20)
#    plt.yticks(fontsize=20)
#    plt.title('%.2f MHz'% (nurange[i]*1000), fontsize=25)
#    filename = 'TruncProfRedo_'+str(i)+'b.png'
#    picpath = "/Users/marisa/Dropbox/MG/DropLaTeX/FirstYearReport/Spyderplots/Raytracing"
#    fileoutput = os.path.join(picpath,filename)
#    plt.obse(fileoutput, dpi=96)
    observedfluxfixed.append(observedfluxfix)

#sys.exit()

observedflux = np.array(observedflux)
freal = observedflux*fluxtr

noiseval = freal[0]/1200.
noise = np.random.normal(0,noiseval,P)

for i in range(len(nurange)):
    observednoise = observedfluxfixed[i] + noise
    observed.append(observednoise)



sys.exit()

    
## Determine the index at which the flux spectrum has a maximum:
idmax = np.argmax(freal)

"""idmax of 8 used before"""
##
#### Fit powerspectrum on both sides of the maximum
#sh1 = 2
#sh2 = 10
#
#idmaxs1 = idmax - sh1
#idmaxs2 = idmax + sh2
#
#modelpow = Model(PLM)
#resultpow1 = modelpow.fit(freal[0:idmaxs1+1],x=nurange[0:idmaxs1+1],K=0.001,k=4)
#resultpow2 = modelpow.fit(freal[idmaxs2:len(freal)],x=nurange[idmaxs2:len(freal)],K=0.001,k=4)
#specfitf1 = resultpow1.best_values['K']
#specfitf2 = resultpow2.best_values['K']
#specfitpos = resultpow1.best_values['k']
#specfitneg = resultpow2.best_values['k']

#parameters = np.array([specfitf1,specfitf2,specfitpos,specfitneg])
#np.savetxt('Pythondata/specfits.txt',parameters);
#np.savetxt('Pythondata/freal.txt',freal);

## Create more interspaced axis only for tickmarks

#new2 = nurange[0:len(freal):2]
#ticksMHz = [40,80,120,160,200, 240, 280, 320,360,400,440]
#
##plt.figure()
#plt.xticks(ticksMHz,ticksMHz,fontsize=18)
#plt.yticks(fontsize=18)
#plt.xlabel(r'$\nu$ (MHz)',fontsize=20)
#plt.ylabel('normalized flux',fontsize=20)
#plt.plot(1000*nurange,freal,'bo',markersize=10)
#plt.plot(1000*nurange[0:idmaxs1+2],specfitf1*np.power(nurange[0:idmaxs1+2],-specfitpos),'r--',linewidth=2.0,label='$\gamma$ = %.2f' %specfitpos)
#plt.plot(1000*nurange[idmax+2:len(freal)],specfitf2*np.power(nurange[idmax+2:len(freal)],-specfitneg),'g--',linewidth=2.0,label='$\gamma$ = %.2f' %specfitneg)
#plt.legend(loc = 'best',fontsize=20)  


sys.exit()
#
#xax = np.arange(0,P,1)
#
#### The models created above (GxFE and GxE) respectively fit for a convolution of 2 Gaussian pulses with a FOLDED and normal exponential. The code will show that GxFE performs best.
###
####Create limits on s, via w50
####The distribution of pulsar duty cylces is heavily skewed with a median at 2.5% and an overall minimum at 0.3% and overall maximum at 63% (Ref:Jayanth)
####This max is clearly huge - and therefore the process is pretty much unconstrained. Should consider inserting the actual distribution
#
#w50min = float((0.3/100)*P)  
#w50max =  float((3.0/100)*P)  
#
#smin = w50min/(2*np.sqrt(2*np.log(2)))
#smax = w50max/(2*np.sqrt(2*np.log(2)))
#
#
#modelname = GxETrain
#model = Model(modelname)
#
#model.set_param_hint('sigma', value=s, vary=True, min=smin, max=smax)
#model.set_param_hint('mu', value=m, vary=True)
#model.set_param_hint('A',value=1.5, vary=True, min=0)
#model.set_param_hint('tau',value=200, vary=True, min=0)
#pars = model.make_params()
##print model.param_hints
#
##modelname2 = GxESingleFold
##model2 = Model(modelname2)
##
##model2.set_param_hint('sigma', value=s, vary=True, min=smin, max=smax)
##model2.set_param_hint('mu', value=m, vary=True)
##model2.set_param_hint('A',value=1.5, vary=True, min=0)
##model2.set_param_hint('tau',value=200, vary=True, min=0)
##pars2 = model2.make_params()
###print model2.param_hints
##
#
#modelpow = Model(PLM)
#
#obtainedtau=[]
#obtainedtaustd=[]
#obtainedA=[]
#obtainedsig=[]
#obtainedmu=[]
#
##obtainedtau2=[]
##obtainedtaustd2=[]
##obtainedA2=[]
##obtainedsig2=[]
##obtainedmu2=[]
##
#
#
#sp = 4 #number of subplots per figure
#numPlots = len(nurange)
#numFigs = int(numPlots/sp)+1
#print numFigs
#
#xsec = np.linspace(0,1.0,P)
#
##for k in range(len(taubins)):
##            print k
#    
#for j in range(1,numFigs+1):    
#    plt.figure(figsize=(14, 11))
##    print j
#    for x in range(1,5):
#        plt.subplot(2,2,x) 
#        plt.subplots_adjust(hspace=.3)
#        k = (j-1)*sp+x-1
#        if k > len(nurange)-1:
#            break
#        else:
#            print j, x, k                      
#            plt.plot(xsec,observed[k]/np.max(observedfluxfixed[0]),'b-')
#          
#### Fit model 1                        
#            result = model.fit(observed[k],pars,x=xax)
#            print(result.fit_report(show_correl = False))
#            besttau = result.best_values['tau']
#            taustd = result.params['tau'].stderr
#            bestA =result.best_values['A'] 
#            bestsig=result.best_values['sigma']
#            bestmu=result.best_values['mu']
#            
#            obtainedtau.append(besttau)  
#            obtainedtaustd.append(taustd)
#            obtainedA.append(bestA) 
#            obtainedsig.append(bestsig) 
#            obtainedmu.append(bestmu)
#            results.append(result)
#                       
#            plt.plot(xsec,result.best_fit/np.max(observedfluxfixed[0]),'r--', linewidth = 2.0,label = r'Fitted $\tau: %.2fs$' % np.divide(obtainedtau[k],P))     
##            plt.plot(xsec,result.best_fit,'r--', linewidth = 2.0,label = r'%s $\tau=%.2f$s' % ('Pre-fold, long', np.divide(obtainedtau[k],P)))     
#
#### Fit model 2            
##            result2 = model2.fit(observed[k],pars2,x=xax)
###            print(result2.fit_report(show_correl = False))
##            besttau2 = result2.best_values['tau']
##            taustd2 = result2.params['tau'].stderr            
##            bestA2 =result2.best_values['A'] 
##            bestsig2=result2.best_values['sigma']
##            bestmu2=result2.best_values['mu']
##            
##            obtainedtau2.append(besttau2)
##            obtainedtaustd2.append(taustd2)
##            obtainedA2.append(bestA2) 
##            obtainedsig2.append(bestsig2) 
##            obtainedmu2.append(bestmu2) 
##            
###            plt.plot(xsec,result2.best_fit/np.max(observed[k]),'b--',linewidth = 2.0, label = r'%s, $\tau=%.2f$' % (modelname2.__name__ , np.divide(obtainedtau2[k],P))) 
##            
#            
#            plt.title(r'%.0f MHz, For $\alpha = 4$: $\tau=%.3fs$ ' % (1000*nurange[k],tauval[k]),fontsize=18) 
#            plt.legend(loc = 'upper right',fontsize=15)
#            plt.xlabel('time (sec)',fontsize=20)
##            plt.ylabel('normalized intensity',fontsize=20)
#            plt.xticks(fontsize=18)
#            plt.yticks(fontsize=18)
#            plt.show()
#             
#            
#            arfit = result.best_fit
#            np.savetxt('Pythondata/RayTrace/arfit%d_tr.txt' %k, arfit)
#            
#            k += 1
#    filename = 'TruncProfiles_'+str(j)+'.png'
#    picpath = "/Users/marisa/Documents/PhD/GitHub_Scattering/Plots/RayTrace"
#    fileoutput = os.path.join(picpath,filename)
#    plt.savefig(fileoutput, dpi=200)
#
#np.savetxt('Pythondata/RayTrace/arN_tr.txt', observed)
#np.savetxt('Pythondata/RayTrace/arNoN_tr.txt', observednonoise)
#np.savetxt('Pythondata/RayTrace/fluxnorm_tr.txt', observedfluxfixed)
#    
#
#obtainedtau = np.array(obtainedtau) 
##obtainedtau2 = np.array(obtainedtau2) 
#
#obtainedtaustd = np.array(obtainedtaustd)
##obtainedtaustd2 = np.array(obtainedtaustd2)
#
#resultpow = modelpow.fit(obtainedtau,x=nurange,K=0.001,k=4)
#specfit = resultpow.best_values['k']
#
##resultpow2 = modelpow.fit(obtainedtau2,x=nurange,K=0.001,k=4)
##specfit2 = resultpow2.best_values['k']
#
#
##taupow4 = psr.tau(Dval,Dsval,kappa1,nurange,light)
##taupow4bins = taupow4*P/pulseperiod
#
## Frequency range as determined by the chosen bandwidth
#
#
#bwhigh = nurange+bandw/2
#bwlow = nurange-bandw/2
#
#rfreq = 10**((np.log10(bwhigh)+ np.log10(bwlow))/2)
#
#bin2sec = pulseperiod/P
#bs= bin2sec
#
#obtainedtauhigh = obtainedtau[0:7]
#obtainedtaulow = obtainedtau[6:17]
#
#reshigh = modelpow.fit(obtainedtauhigh,x=nurange[0:7],K=0.001,k=4)
#reslow = modelpow.fit(obtainedtaulow,x=nurange[6:17],K=0.001,k=4)
#
#specfithigh = reshigh.best_values['k']
#specfitlow = reslow.best_values['k']
##ticksMHz = [40,60,80,100,120,140,160,180,200,220,240]
#ticksMHz = [40,60,80,100,120,140,160,180,200]
#
#plt.figure(figsize=(14, 9))
#plt.errorbar(1000*nurange,obtainedtau*bs,yerr=obtainedtaustd*bs,fmt='ro',markersize=8, capthick=2,linewidth=1.5, label = r'%s, $\alpha=$ %.2f' % ('Pre-fold' , specfit))
#plt.plot(1000*nurange,resultpow.best_fit*bs,'r--',linewidth = 2.0,)
#plt.plot(1000*nurange[0:7],reshigh.best_fit*bs,'b-.',linewidth = 2.0,label = r'$\alpha=$ %.2f' % specfithigh)
#plt.plot(1000*nurange[6:17],reslow.best_fit*bs,'g-.',linewidth = 2.0,label = r'$\alpha=$ %.2f' %  specfitlow)
#plt.plot(1000*nurange,taubins*bs,'m*-',markersize=10.0,linewidth=2.0,label = r'$\alpha= 4.00$')
#plt.xlim(1000*(nulow-0.01),1000*(nuhigh+0.01))
#plt.xlabel('frequency (MHz)', fontsize=30)
#plt.ylabel(r'$\tau$ (sec)',fontsize=30)
#plt.legend(loc ='best',fontsize=22)
#plt.xscale('log')
#plt.yscale('log')
#plt.xticks(ticksMHz,ticksMHz,fontsize=26)
#plt.yticks(fontsize=26)
#
#filename2 = 'TruncTau.png'
#picpath = "/Users/marisa/Documents/PhD/GitHub_Scattering/Plots/RayTrace"
#fileoutput = os.path.join(picpath,filename2)
#plt.savefig(fileoutput, dpi=200)
#
#
#reshighfit = reshigh.best_fit
#reslowfit = reslow.best_fit
#resonefit = resultpow.best_fit
#
#np.savetxt('Pythondata/RayTrace/tau.txt', obtainedtau)
#np.savetxt('Pythondata/RayTrace/tauhigh.txt', obtainedtauhigh)
#np.savetxt('Pythondata/RayTrace/taulow.txt', obtainedtaulow)
#np.savetxt('Pythondata/RayTrace/taustd.txt', obtainedtaustd)
#np.savetxt('Pythondata/RayTrace/fithigh.txt', reshighfit)
#np.savetxt('Pythondata/RayTrace/fitlow.txt', reslowfit)
#np.savetxt('Pythondata/RayTrace/fitall.txt', resonefit)
#np.savetxt('Pythondata/RayTrace/taubins.txt', taubins)
#
#
#
#print("--- %s seconds ---" % (time.time() - starttime))
#

#sys.exit()
    
# AniIsoFluxPlot
    
#sh1 = 4
#sh2 = 10
#
#"""idmax of 5 found before"""
#
#idmaxs1 = idmax - sh1
#idmaxs2 = idmax +sh2
#
#modelpow = Model(PLM)
#resultpow1 = modelpow.fit(freal[0:idmaxs1+1],x=nurange[0:idmaxs1+1],K=0.001,k=4)
#resultpow2 = modelpow.fit(freal[idmaxs2:len(freal)],x=nurange[idmaxs2:len(freal)],K=0.001,k=4)
#specfitf1 = resultpow1.best_values['K']
#specfitf2 = resultpow2.best_values['K']
#specfitpos = resultpow1.best_values['k']
#specfitneg = resultpow2.best_values['k']
#
#parametersA = np.array([specfitf1,specfitf2,specfitpos,specfitneg])
#np.savetxt('Pythondata/specfits_ani.txt',parametersA);
#np.savetxt('Pythondata/freal_ani.txt',freal)
#
### Create more interspaced axis only for tickmarks
#
##new2 = newnurange[0:len(freal):2]
#ticksMHz = [40,80,120,160,200, 240, 280, 320,360,400,440]
#
#plt.figure()
#plt.xticks(ticksMHz,ticksMHz,fontsize=18)
#plt.yticks(fontsize=18)
#plt.xlabel(r'$\nu$ (MHz)',fontsize=20)
#plt.ylabel('normalized flux',fontsize=20)
#plt.plot(1000*nurange,freal,'mo',markersize=10)
#plt.plot(1000*nurange[0:idmax-1],specfitf1*np.power(nurange[0:idmax-1],-specfitpos),'c--',linewidth=2.0,label=r'$\gamma$ = %.2f' %specfitpos)
#plt.plot(1000*nurange[idmax+3:len(freal)],specfitf2*np.power(nurange[idmax+3:len(freal)],-specfitneg),'y--',linewidth=2.0,label=r'$\gamma$ = %.2f' %specfitneg)
#plt.legend(loc = 'best',fontsize=20)
#
