import pypsr as psr
import os, sys
#import matplotlib as mp
import matplotlib.pyplot as plt
from lmfit import Model
from math import pi
#from lmfit import minimize, Parameter, Parameters, fit_report
import numpy as np
#from scipy import special


def PLM(x,K,k):
    return K*pow(x,-k)


### Universal constants 
light = 9.71561189*10**-12      #Speed of light in kpc/s  
mrad = 4.85*10**-9              #Convertion from miliarcseconds to radians

### Input the observed frequency range

nulow, nuhigh = float(0.04),float(0.22)


### Input the bins resolution

nbinspow = 9
P = 2**nbinspow

### Create the properties of the ingoing pulse

pulseperiod =1.0                  #in seconds
dutycycle = float(5)             #as a % of the overall pulseperiod
m = float(P/4)                      #Let the mean of the pulse be at a 1/4 overall bins
w50 = float((dutycycle/100)*P)      #FWHM
s = w50/(2*np.sqrt(2*np.log(2)))    #sigma calculated through the dutycycle
a = 1                               #amplitude. at some point this will be replaced by the intrinsic spectrum. 

trainlength = 20
snr = 10

#plt.figure(figsize=(14, 9))
#fig = plt.figure(figsize=(9, 9))
#for j in range(0,4):
#    Dval = float(3.0 + 1*j)
##    Dval = float(3.0 + 3*j)
#    for i in range(0,5):
#    #    Dval, Dsval = float(3.0),float(0.3)+0.3*i
#        Dsval = float(Dval/10)+(Dval/10)*i
##        for k1 in range(3,15,3):
Dval, Dsval = float(3.0), float(1.5)    

k1 = 3
k2 = 1
kappa1 = k1*mrad
kappa2 = k2*mrad

nurange = np.arange(nulow,nuhigh,0.001)


tauval = psr.tau(Dval,Dsval,kappa1,nurange,light)
    
#if tauval[0] > 5*pulseperiod:
#    sys.exit("tau1 warning: scattering is too high")
#else:
#    taubins = (tauval/pulseperiod)*P
taubins = (tauval/pulseperiod)*P

tauval2 = psr.tau(Dval,Dsval,kappa2,nurange,light)

#if tauval2[0] > 5*pulseperiod:
#    sys.exit("tau2 warning: scattering is too high")
#else:
#    taubins2 = (tauval2/pulseperiod)*P

taubins2 = (tauval2/pulseperiod)*P

#print taubins

bins, profile = psr.makeprofile(nbins = P, ncomps = 1, amps = a, means = m, sigmas = s)

spectralindex = 1.6  #Input spectral index as a postivie number, for nu^-alpha
profile_intr = psr.profilespec(nurange,spectralindex,profile)
profile_intr_norm = profile_intr/np.sum(profile_intr[0])   #Nomralised such that the max intrinsic flux = 1.0. That is the intrinsice pulse at the lowest frequency has flux = 1

#sys.exit()

###############################################################################
## ISOTROPIC
###############################################################################

##Create observed pulses by scattering a pulsetrain with an exponential broadening function
  
observed = []
observednonoise = []
observedflux = []
scatt = []
#
#        plt.subplot(2,2,j+1) 
##        plt.subplot(2,1,j+1) 
#        plt.subplots_adjust(hspace=.2)
#            
for i in range(0,len(taubins)):
    scat = psr.psrscatter(bins, psr.broadfunc(bins,taubins[i]),psr.pulsetrain(trainlength, bins, profile_intr_norm[i]),taubins[i])
    scatt.append(scat)
    climb, observed_nonoise, flux = psr.extractpulse(scat, 2, P)
    peak = np.max(observed_nonoise)
    noise = np.random.normal(0,peak/snr,P)
    observedadd = observed_nonoise + noise
    observed.append(observedadd)
    observedflux.append(flux)
    observednonoise.append(observed_nonoise)
#        #    plt.plot(nurange,observedflux,linewidth=2.0,label=r'$\sigma_{x,y} = %d$ mas. $\tau_{max} = %.1f$' % (k1, tauval[0]))
##        plt.plot(1000*nurange,observedflux,linewidth=2.0,label=r'$D_s/D = %.1f$, $\tau_{max} = %.1f$' % (Dsval/Dval,tauval[0]))    
#        plt.plot(1000*nurange,observedflux,linewidth=2.0,label=r'$D_s/D = %.1f$' % (Dsval/Dval))    
#        plt.title('D = %.0f kpc' %Dval)
#        plt.legend(loc = 'best',fontsize=14)
##        plt.xlabel(r'$\nu$ (MHz)',fontsize=16)
#        plt.ylabel('normalized flux',fontsize=16)
#        plt.xticks(fontsize=14)
#        plt.yticks(fontsize=14)
#        plt.xlim(40,250)
#        
## Set common x-label
#plt.text(145, -0.05, r'$\nu$ (MHz)', ha='center', va='center', fontsize = 16)
#    
#
#        #plt.ylim(0,0.4)
#        #plt.title('Flux Spectrum - Isotropic')
#            

#sys.exit()

#xax = np.arange(P,2*P,1)

## The models created above (GxFE and GxE) respectively fit for a convolution of 2 Gaussian pulses with a FOLDED and normal exponential. The code will show that GxFE performs best.
#
##Create limits on s, via w50
##The distribution of pulsar duty cylces is heavily skewed with a median at 2.5% and an overall minimum at 0.3% and overall maximum at 63% (Ref:Jayanth)
##This max is clearly huge - and therefore the process is pretty much unconstrained. Should consider inserting the actual distribution

w50min = float((0.3/100)*P)  
w50max =  float((3.0/100)*P)  

smin = w50min/(2*np.sqrt(2*np.log(2)))
smax = w50max/(2*np.sqrt(2*np.log(2)))

max1 = np.max(taubins)
max2 = np.max(taubins2)
maxall = np.max((max1,max2))


modelpow = Model(PLM)

#
## Now plot the flux spectra
skip = 9

maxindex = np.array(observedflux).argmax()
obspos = observedflux[0:maxindex-skip]
obsneg = observedflux[maxindex+skip:len(observedflux)]

respos = modelpow.fit(obspos,x=nurange[0:maxindex-skip],K=1.0,k=-2)
resneg = modelpow.fit(obsneg,x=nurange[maxindex+skip:len(observedflux)],K=1.0,k=2)

posalf = respos.best_values['k']
posK = respos.best_values['K']
negalf = resneg.best_values['k']
negK = resneg.best_values['K']

#plt.figure()
#plt.plot(nurange,observedflux,'b',linewidth = 2.5)
#plt.plot(nurange[0:maxindex],respos.best_fit,'g--',linewidth = 2.0, label=r'$\alpha_{iso}$ = %.1f' % -respos.best_values['k'])
#plt.plot(nurange[maxindex+skip:len(observedflux)],resneg.best_fit,'c--',linewidth = 2.0, label=r'$\alpha_{iso}$ = %.1f' % -resneg.best_values['k'])
##plt.title('Flux spectrum')
#plt.xlabel(r"$\nu$ (GHz)")
#plt.ylabel("flux (normalised to a max intrinsic flux of 1.0)")
#plt.legend(loc = 'best')
##plt.xlim(0.05,0.3)
##plt.ylim(0,0.3)



###############################################################################
###############################################################################
###############################################################################
##                                                                           ##
##     ANISOTROPIC SCATTERING AND FITTING                                    ##
##                                                                           ##
###############################################################################
###############################################################################
###############################################################################       

observedAni = []
observedfluxAni = []
scatAni=[]

for i in range(0,len(taubins)):
    scatA = psr.psrscatter(bins, psr.broadfunc2(bins,taubins[i], taubins2[i]),psr.pulsetrain(trainlength, bins, profile_intr_norm[i]),taubins[i])
    scatAni.append(scatA)    
    climbA, observed_nonoiseA, fluxA = psr.extractpulse(scatA, 2, P)
    peak = np.max(observed_nonoiseA)
    noise = np.random.normal(0,peak/snr,P)
    observedAniadd = observed_nonoiseA + noise
    observedAni.append(observedAniadd)
    observedfluxAni.append(fluxA)
##plt.plot(nurange,observedfluxAni,'--',label=r'$\sigma_x = %d$ mas, $\sigma_y = %d$ mas' % (k1,k2))
##plt.legend(loc = 'best',fontsize=8.0)
##plt.xlabel(r'$\nu$ (GHz)')
##plt.ylabel('flux')
##plt.title('Flux Spectrum - Isotropic and Anisotropic')
#
#
#max1 = np.max(taubins)
#max2 = np.max(taubins2)
#maxall = np.max((max1,max2))

### Now plot the flux spectra


skipA = 20
skipA2 = 10

maxindexAni = np.array(observedfluxAni[0:150]).argmax()
obsposAni = observedfluxAni[0:maxindexAni-skipA2]
obsnegAni = observedfluxAni[maxindexAni+skipA:len(observedfluxAni)]

resposAni = modelpow.fit(obsposAni,x=nurange[0:maxindexAni-skipA2],K=1.0,k=-2)
resnegAni = modelpow.fit(obsnegAni,x=nurange[maxindexAni+skipA:len(observedfluxAni)],K=1.0,k=2)
posanialf = resposAni.best_values['k']
posaniK = resposAni.best_values['K']
neganialf = resnegAni.best_values['k']
neganiK = resnegAni.best_values['K']

nurangeMHz = nurange*1000

plt.figure(figsize=(9,7))
plt.plot(nurangeMHz,observedfluxAni,'m-',linewidth = 2.5)
#plt.plot(nurange[0:maxindexAni-skipA2],resposAni.best_fit,'m-.',linewidth = 2.0, label=r'$\alpha_{aniso}$ = %.1f' % -resposAni.best_values['k'])
plt.plot(nurangeMHz[0:maxindexAni-skipA2+10],posaniK*np.power(nurange[0:maxindexAni-skipA2+10],-posanialf),'r-.',linewidth = 2.0, label=r'$\gamma_{aniso}$ = %.1f' % resposAni.best_values['k'])
#plt.plot(nurangeMHz[maxindexAni+skipA:len(observedfluxAni)],resnegAni.best_fit,'k-.',linewidth = 2.0, label=r'$\alpha_{aniso}$ = %.1f' % resnegAni.best_values['k'])
plt.plot(nurangeMHz[maxindexAni+8:len(observedfluxAni)],neganiK*np.power(nurange[maxindexAni+8:len(observedfluxAni)],-neganialf),'k-.',linewidth = 2.0, label=r'$\gamma_{aniso}$ = %.1f' % resnegAni.best_values['k'])
plt.text(80,0.32,r'$\sigma_x = %d$mas, $\sigma_y = %d$mas at 1 GHz ' % (k1,k2), fontsize=15)
plt.text(120,0.18,r'$\sigma_{x,y} = %d$mas at 1 GHz ' % k1, fontsize=15)
plt.text(120,0.21,'isotropic',fontsize=18)
plt.text(80,0.35,'anisotropic',fontsize=18)


#plt.figure()
plt.plot(nurangeMHz,observedflux,'c',linewidth = 2.5)
#plt.plot(nurangeMHz[0:maxindex-skip],respos.best_fit,'g--',linewidth = 2.0, label=r'$\alpha_{iso}$ = %.1f' % respos.best_values['k'])
plt.plot(nurangeMHz[0:maxindex-4],posK*np.power(nurange[0:maxindex-4],-posalf),'g--',linewidth = 2.0, label=r'$\gamma_{iso}$ = %.1f' % respos.best_values['k'])
#plt.plot(nurangeMHz[maxindex+skip:len(observedflux)],resneg.best_fit,'b--',linewidth = 2.0, label=r'$\alpha_{iso}$ = %.1f' % resneg.best_values['k'])
plt.plot(nurangeMHz[maxindex-3:len(observedflux)],negK*np.power(nurange[maxindex-3:len(observedflux)],-negalf),'b--',linewidth = 2.0, label=r'$\gamma_{iso}$ = %.1f' % resneg.best_values['k'])
plt.xlabel(r"$\nu$ (MHz)",fontsize=20)
plt.ylabel("normalized flux", fontsize=20)
plt.legend(loc = 'best',fontsize=16)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.xlim(40,220)
#plt.ylim(0,0.3)

#filename = 'FluxAnisoMHz.png'
#picpath = "/Users/marisa/Dropbox/Aris/TexOutputs/ScatteringPaperTexAndImages/Spyderplots"
#fileoutput = os.path.join(picpath,filename)
#plt.savefig(fileoutput, dpi=200)
