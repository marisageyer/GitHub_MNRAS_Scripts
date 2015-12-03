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
from ImportDatabaseProfile import epnprofy
####

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
 
def GxETrainLong(x,mu,sigma, A, tau):
    #This model convolves a pulsetrain with a broadening function
    #It extracts one of the last convolved profiles, subtracts the climbed baseline and then adds noise to it
    trainlength = 10    
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

pulseperiod =1.382                  #in seconds
dutycycle = float(2.5)             #as a % of the overall pulseperiod

## Create the profile to have the same time resolution than the broadening function created by photonscat.py

P = int(pulseperiod/binstimeres)

specepn = 1.8
epn_intr = psr.profilespec(nurange,specepn,epnprofy)
epn_norm = epn_intr/np.sum(epn_intr[0])

## Resample epn_norm to have the same time resolution than the broadening function produced by Ray_minimal.py
newbinnumber = pulseperiod/binstimeres
epnbintimeres = np.linspace(0.,1.0,len(epn_norm[0]))
newtimeres= np.linspace(0,1.0,newbinnumber)
epn_norm_interpp = []
for k in range(len(epn_norm)):
    epn_norm_interp = np.interp(newtimeres,epnbintimeres,epn_norm[k])
    epn_norm_interpp.append(epn_norm_interp)



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


xsec = np.linspace(0,pulseperiod,P)



#np.savetxt('Pythondata/photonsfreq.txt',photonsfreq)

for i in range(len(nurange)):
#    scat = psr.psrscatter(probfreq[i],psr.pulsetrain(3, bins, profile_intr_norm[i]))
#    scat = psr.psrscatter(photonsinterpp[i],epn_norm[i])
    scat = psr.psrscatter(probfreq[i],epn_norm_interpp[i])
    scatt.append(scat)      
#    climb, observed_nonoise, rec, flux = psr.extractpulse(scat, 2, P)
    climb, observed_nonoise, rec, flux = psr.extractpulse(scat, 0, P)
#    observedflux.append(flux)
    observednonoise.append(observed_nonoise)
    
np.savetxt("Pythondata/EPN123725/Profiles_120_220.txt",observednonoise)    

#sys.exit()
# the flux as produced by psr.extractpulse is normalised such that the lowest frequency has flux 1
# Profiles:
#plt.figure()
for i in range(len(nurange)):    
#    observedfluxfix = observednonoise[i]*fluxtr[i]
    plt.figure()    
#    plt.plot(xsec,observedfluxfix/np.max(observedfluxfix),'b',linewidth = 4.0)
#    plt.plot(xsec,scatt[i]/np.max(scatt[i]),'m',linewidth = 4.0)
    plt.plot(xsec,observednonoise[i]/np.max(observednonoise[i]),'g',linewidth = 4.0)
    plt.xlabel("time(sec)",fontsize=20)
    plt.ylabel("normalized intensity",fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.title('%.2f MHz'% (nurange[i]*1000), fontsize=25)
#    filename = 'TruncProfRedo_'+str(i)+'b.png'
#    picpath = "/Users/marisa/Dropbox/MG/DropLaTeX/FirstYearReport/Spyderplots/Raytracing"
#    fileoutput = os.path.join(picpath,filename)
#    plt.savefig(fileoutput, dpi=96)
#    observedfluxfixed.append(observedfluxfix)

sys.exit()

observedflux = np.array(observedflux)
freal = observedflux*fluxtr

noiseval = freal[0]/1200.
noise = np.random.normal(0,noiseval,P)



for i in range(len(nurange)):
    observednoise = observedfluxfixed[i] + noise
    observed.append(observednoise)



#sys.exit()

    
## Determine the index at which the flux spectrum has a maximum:
idmax = np.argmax(freal)

## Fit powerspectrum on both sides of the maximum
sh1 = 0
sh2 = 0

idmaxs1 = idmax - sh1
idmaxs2 = idmax + sh2

modelpow = Model(PLM)
resultpow1 = modelpow.fit(freal[0:idmaxs1+1],x=nurange[0:idmaxs1+1],K=0.001,k=4)
resultpow2 = modelpow.fit(freal[idmaxs2:len(freal)],x=nurange[idmaxs2:len(freal)],K=0.001,k=4)
specfitf1 = resultpow1.best_values['K']
specfitf2 = resultpow2.best_values['K']
specfitpos = resultpow1.best_values['k']
specfitneg = resultpow2.best_values['k']

## Create more interspaced axis only for tickmarks

#new2 = newnurange[0:len(freal):2]
ticksMHz = [40,80,120,160,200, 240, 280, 320,360,400,440]

plt.figure()
plt.xticks(ticksMHz,ticksMHz,fontsize=18)
plt.yticks(fontsize=18)
plt.xlabel(r'$\nu$ (MHz)',fontsize=20)
plt.ylabel('normalized flux',fontsize=20)
plt.plot(1000*nurange,freal,'bo',markersize=10)
plt.plot(1000*nurange[0:idmaxs1+1],specfitf1*np.power(nurange[0:idmaxs1+1],-specfitpos),'r--',linewidth=2.0,label=r'$\gamma$ = %.2f' %-specfitpos)
plt.plot(1000*nurange[idmax:len(freal)],specfitf2*np.power(nurange[idmax:len(freal)],-specfitneg),'g--',linewidth=2.0,label=r'$\gamma$ = %.2f' %-specfitneg)
plt.legend(loc = 'best',fontsize=16)  


sys.exit()

#for lf in range(20,25,10)


lf =30
    
print 'lf: '+str(lf)       

xax = np.arange(0,P,1)

### The models created above (GxFE and GxE) respectively fit for a convolution of 2 Gaussian pulses with a FOLDED and normal exponential. The code will show that GxFE performs best.
##
###Create limits on s, via w50
###The distribution of pulsar duty cylces is heavily skewed with a median at 2.5% and an overall minimum at 0.3% and overall maximum at 63% (Ref:Jayanth)
###This max is clearly huge - and therefore the process is pretty much unconstrained. Should consider inserting the actual distribution

w50min = float((0.3/100)*P)  
w50max =  float((3.0/100)*P)  

smin = w50min/(2*np.sqrt(2*np.log(2)))
smax = w50max/(2*np.sqrt(2*np.log(2)))


modelname = GxETrain
model = Model(modelname)

model.set_param_hint('sigma', value=s, vary=True, min=smin, max=smax)
model.set_param_hint('mu', value=m, vary=True)
model.set_param_hint('A',value=1.5, vary=True, min=0)
model.set_param_hint('tau',value=200, vary=True, min=0)
pars = model.make_params()
#print model.param_hints

#modelname2 = GxESingleFold
#model2 = Model(modelname2)
#
#model2.set_param_hint('sigma', value=s, vary=True, min=smin, max=smax)
#model2.set_param_hint('mu', value=m, vary=True)
#model2.set_param_hint('A',value=1.5, vary=True, min=0)
#model2.set_param_hint('tau',value=200, vary=True, min=0)
#pars2 = model2.make_params()
##print model2.param_hints
#
#modelname3 = GxFE
#model3 = Model(modelname3)
#
#model3.set_param_hint('sigma', value=s, vary=False, min=smin, max=smax)
#model3.set_param_hint('mu', value=m, vary=False)
#model3.set_param_hint('A',value=1.5, vary=True, min=0)
#model3.set_param_hint('tau',value=200, vary=True, min=0)
#pars3 = model3.make_params()
#
#xax3 = np.arange(P,2*P,1)
#
#
#modelname4 = GxETrainShort
#model4 = Model(modelname4)
#
#model4.set_param_hint('sigma', value=s, vary=False, min=smin, max=smax)
#model4.set_param_hint('mu', value=m, vary=False)
#model4.set_param_hint('A',value=1.5, vary=True, min=0)
#model4.set_param_hint('tau',value=200, vary=True, min=0)
#pars4 = model4.make_params()
#
##print model2.param_hints
#

modelpow = Model(PLM)

obtainedtau=[]
obtainedtaustd=[]
obtainedA=[]
obtainedsig=[]
obtainedmu=[]

#obtainedtau2=[]
#obtainedtaustd2=[]
#obtainedA2=[]
#obtainedsig2=[]
#obtainedmu2=[]
#
#obtainedtau3=[]
#obtainedtaustd3 = []
#obtainedA3=[]
#obtainedsig3=[]
#obtainedmu3=[]
#
#obtainedtau4=[]
#obtainedtaustd4 = []
#obtainedA4=[]
#obtainedsig4=[]
#obtainedmu4=[]
#


#tau_err_high = []
#tau_err_low = []
#tau_err_high2 = []
#tau_err_low2 = []
#tau_err_high3 = []
#tau_err_low3 = []


sp = 4 #number of subplots per figure
numPlots = len(nurange)
numFigs = int(numPlots/sp)+1
print numFigs

xsec = np.linspace(0,1.0,P)

#for k in range(len(taubins)):
#            print k
    
for j in range(1,numFigs+1):    
    plt.figure(figsize=(14, 11))
#    print j
    for x in range(1,5):
        plt.subplot(2,2,x) 
        plt.subplots_adjust(hspace=.3)
        k = (j-1)*sp+x-1
        if k > len(nurange)-1:
            break
        else:
            print j, x, k                      
            plt.plot(xsec,observed[k]/np.max(observedfluxfixed[0]),'b-')
          
### Fit model 1                        
            result = model.fit(observed[k],pars,x=xax)
            print(result.fit_report(show_correl = False))
            besttau = result.best_values['tau']
            taustd = result.params['tau'].stderr
            bestA =result.best_values['A'] 
            bestsig=result.best_values['sigma']
            bestmu=result.best_values['mu']
            
            obtainedtau.append(besttau)  
            obtainedtaustd.append(taustd)
            obtainedA.append(bestA) 
            obtainedsig.append(bestsig) 
            obtainedmu.append(bestmu)
            
#            
#            
##            ciFE = conf_interval(result,sigmas=[0.997])
##            tau_err_h = ciFE['tau'][4][1]-ciFE['tau'][2][1]
##            tau_err_l=ciFE['tau'][2][1]-ciFE['tau'][0][1]
##            tau_err_high.append(tau_err_h)
##            tau_err_low.append(tau_err_l)
####            printfuncs.report_ci(ciFE)
#            
            plt.plot(xsec,result.best_fit/np.max(observedfluxfixed[0]),'r--', linewidth = 2.0,label = r'Fitted $\tau: %.2fs$' % np.divide(obtainedtau[k],P))     
#            plt.plot(xsec,result.best_fit,'r--', linewidth = 2.0,label = r'%s $\tau=%.2f$s' % ('Pre-fold, long', np.divide(obtainedtau[k],P)))     

### Fit model 2            
#            result2 = model2.fit(observed[k],pars2,x=xax)
##            print(result2.fit_report(show_correl = False))
#            besttau2 = result2.best_values['tau']
#            taustd2 = result2.params['tau'].stderr            
#            bestA2 =result2.best_values['A'] 
#            bestsig2=result2.best_values['sigma']
#            bestmu2=result2.best_values['mu']
#            
#            obtainedtau2.append(besttau2)
#            obtainedtaustd2.append(taustd2)
#            obtainedA2.append(bestA2) 
#            obtainedsig2.append(bestsig2) 
#            obtainedmu2.append(bestmu2) 
#            
##            ciFE2 = conf_interval(result2,sigmas=[0.997])
##            tau_err_h2 = ciFE2['tau'][4][1]-ciFE2['tau'][2][1]
##            tau_err_l2=ciFE2['tau'][2][1]-ciFE2['tau'][0][1]
##            tau_err_high2.append(tau_err_h2)
##            tau_err_low2.append(tau_err_l2)            
#            
##            plt.plot(xsec,result2.best_fit/np.max(observed[k]),'b--',linewidth = 2.0, label = r'%s, $\tau=%.2f$' % (modelname2.__name__ , np.divide(obtainedtau2[k],P))) 
#            
#### Fit model 3            
#            result3 = model3.fit(observed[k],pars3,x=xax3)
# #           print(result2.fit_report(show_correl = False))
#            besttau3 = result3.best_values['tau']
#            taustd3 = result3.params['tau'].stderr  
#            bestA3 =result3.best_values['A'] 
#            bestsig3=result3.best_values['sigma']
#            bestmu3=result3.best_values['mu']
#            
#            obtainedtau3.append(besttau3)
#            obtainedtaustd3.append(taustd3)
#            obtainedA3.append(bestA3) 
#            obtainedsig3.append(bestsig3) 
#            obtainedmu3.append(bestmu3) 
##            
##            ciFE3 = conf_interval(result3,sigmas=[0.95,0.997])
##            tau_err_h3 = ciFE3['tau'][4][1]-ciFE3['tau'][2][1]
##            tau_err_l3=ciFE3['tau'][2][1]-ciFE3['tau'][0][1]
##            tau_err_high3.append(tau_err_h3)
##            tau_err_low3.append(tau_err_l3)            
#            
##            plt.plot(xsec,result3.best_fit/np.max(observed[k]),'g--',linewidth = 2.0, label = r'%s, $\tau=%.2f$' % (modelname3.__name__ , np.divide(obtainedtau3[k],P))) 
#              
#            
#            result4 = model4.fit(observed[k],pars4,x=xax3)
##            print(result2.fit_report(show_correl = False))
#            besttau4 = result4.best_values['tau']
#            taustd4 = result4.params['tau'].stderr            
#            bestA4 =result4.best_values['A'] 
#            bestsig4=result4.best_values['sigma']
#            bestmu4=result4.best_values['mu']
#            
#            obtainedtau4.append(besttau4)
#            obtainedtaustd4.append(taustd4)
#            obtainedA4.append(bestA4) 
#            obtainedsig4.append(bestsig4) 
#            obtainedmu4.append(bestmu4) 
#            
##            plt.plot(xsec,result4.best_fit/np.max(observed[k]),'c--',linewidth = 2.0, label = r'%s, $\tau=%.2f$' % (modelname4.__name__ , np.divide(obtainedtau4[k],P))) 
#            
#            
            plt.title(r'%.0f MHz, For $\alpha = 4$: $\tau=%.3fs$ ' % (1000*nurange[k],tauval[k]),fontsize=12) 
            plt.legend(loc = 'best',fontsize=11)
            plt.xlabel('time (sec)',fontsize=12)
            plt.ylabel('normalized intensity',fontsize=12)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.show()
            k += 1
##    filename = 'IsoProfiles_'+str(j)+'.png'
##    picpath = "/Users/marisa/Documents/PhD/GitHub_Scattering/Plots"
##    fileoutput = os.path.join(picpath,filename)
##    plt.savefig(fileoutput, dpi=96)
#
obtainedtau = np.array(obtainedtau) 
#obtainedtau2 = np.array(obtainedtau2) 
#obtainedtau3 = np.array(obtainedtau3) 
#obtainedtau4 = np.array(obtainedtau4) 

obtainedtaustd = np.array(obtainedtaustd)
#obtainedtaustd2 = np.array(obtainedtaustd2)
#obtainedtaustd3 = np.array(obtainedtaustd3)
#obtainedtaustd4 = np.array(obtainedtaustd4) 

resultpow = modelpow.fit(obtainedtau,x=nurange,K=0.001,k=4)
specfit = resultpow.best_values['k']

#resultpow2 = modelpow.fit(obtainedtau2,x=nurange,K=0.001,k=4)
#specfit2 = resultpow2.best_values['k']
#
#resultpow3 = modelpow.fit(obtainedtau3,x=nurange,K=0.001,k=4)
#specfit3 = resultpow3.best_values['k']
#
#resultpow4 = modelpow.fit(obtainedtau4,x=nurange,K=0.001,k=4)
#specfit4 = resultpow4.best_values['k']

#taupow4 = psr.tau(Dval,Dsval,kappa1,nurange,light)
#taupow4bins = taupow4*P/pulseperiod



toperr = obtainedtau + obtainedtaustd
bottomerr = obtainedtau - obtainedtaustd
#toperr2 = obtainedtau2 + obtainedtaustd2
#bottomerr2 = obtainedtau2 - obtainedtaustd2
#toperr3 = obtainedtau3 + obtainedtaustd3
#bottomerr3 = obtainedtau3 - obtainedtaustd3
#toperr4 = obtainedtau4 + obtainedtaustd4
#bottomerr4 = obtainedtau4 - obtainedtaustd4
# Frequency range as determined by the chosen bandwidth


bwhigh = nurange+bandw/2
bwlow = nurange-bandw/2

rfreq = 10**((np.log10(bwhigh)+ np.log10(bwlow))/2)

bin2sec = pulseperiod/P
bs= bin2sec

obtainedtauhigh = obtainedtau[0:7]
obtainedtaulow = obtainedtau[6:13]

reshigh = modelpow.fit(obtainedtauhigh,x=nurange[0:7],K=0.001,k=4)
reslow = modelpow.fit(obtainedtaulow,x=nurange[6:13],K=0.001,k=4)

specfithigh = reshigh.best_values['k']
specfitlow = reslow.best_values['k']
ticksMHz = [80,90,100,110,120,130,140,150,160,170,180,190,200]

plt.figure(figsize=(14, 9))
plt.errorbar(1000*rfreq,obtainedtau*bs,yerr=[bottomerr*bs,toperr*bs],fmt='ro',markersize=8, capthick=2,linewidth=1.5, label = r'%s, $\alpha=$ %.2f' % ('Pre-fold, long' , specfit))
plt.plot(1000*rfreq,resultpow.best_fit*bs,'r--',linewidth = 2.0,)
plt.plot(1000*rfreq[0:7],reshigh.best_fit*bs,'b-.',linewidth = 2.0,label = r'$\alpha=$ %.2f' % specfithigh)
plt.plot(1000*rfreq[6:13],reslow.best_fit*bs,'g-.',linewidth = 2.0,label = r'$\alpha=$ %.2f' %  specfitlow)
plt.plot(1000*rfreq,taubins*bs,'m*-',markersize=10.0,linewidth=2.0,label = r'$\alpha= 4.00$')
plt.xlim(1000*(rfreq[0]-0.01),1000*(nuhigh+0.01))
plt.xlabel('frequency (MHz)', fontsize=26)
plt.ylabel(r'$\tau$ (sec)',fontsize=26)
plt.legend(loc ='best',fontsize=18)
plt.xscale('log')
plt.yscale('log')
plt.xticks(ticksMHz,ticksMHz,fontsize=20)
plt.yticks(fontsize=20)



print("--- %s seconds ---" % (time.time() - starttime))


#sys.exit()
    
# AniIsoFluxPlot
    
#sh1 = 4
#sh2 = 10
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
#plt.plot(1000*nurange[0:idmax-1],specfitf1*np.power(nurange[0:idmax-1],-specfitpos),'c--',linewidth=2.0,label=r'$\gamma$ = %.2f' %-specfitpos)
#plt.plot(1000*nurange[idmax+3:len(freal)],specfitf2*np.power(nurange[idmax+3:len(freal)],-specfitneg),'y--',linewidth=2.0,label=r'$\gamma$ = %.2f' %-specfitneg)
#plt.legend(loc = 'best',fontsize=16)

