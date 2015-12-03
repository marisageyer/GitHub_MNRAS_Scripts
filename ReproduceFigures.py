# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 09:25:31 2015

@author: marisa
"""
import numpy as np
import matplotlib.pyplot as plt
import pypsr_redo as psr
from lmfit import Model
import os

""" This script takes the saved .txt files from Phython data produced by
generate_observed_comparemodels_baseline_nolf.py, to reproduce the
profile model fits and tau spectra (from the article)"""

"""Reenter some of the easily obtained values:"""
#(Ideally make header file with all this info)
light = 9.71561189*10**-12 
mrad = 4.85*10**-9     
incr = 0.005
nulow, nuhigh = float(0.06),float(0.125)
#nulow, nuhigh = float(0.09),float(0.155)
#nulow, nuhigh = float(0.15),float(0.215)
#nulow, nuhigh = float(0.20),float(0.265)
nurange = np.arange(nulow,nuhigh,incr)
pulseperiod = 1.0
bins = 512
bs = pulseperiod/bins
Dval = 3.0
Dsval = 1.5
k1 = 3.0
kappa1 = k1*mrad

taupow4 = psr.tau(Dval,Dsval,kappa1,nurange,light)
taupow4bins = taupow4*bins/pulseperiod

"""Create a dictionaries of all the variable names which are imported back"""
"""Dictionary for profiles and model fits"""
timestring = '20151105'
#folder = 'SlowPulsar'
folder = 'Aniso'
#string2 = 'HF'
string2 = '20151108-00'
d = {} 

"""Choose arsec.txt for 1.0 sec pulsar and arsec_B.txt for milisecond pulsar."""

d["arsec"]  =  np.loadtxt('Pythondata/arsec/arsec.txt')

for k in range(len(nurange)):    
    d["arN_{0}".format(k)]  =  np.loadtxt('Pythondata/arN/%s/arN_%d_%s.txt' %(folder,k,string2))
    d["arNoN_{0}".format(k)]  =  np.loadtxt('Pythondata/arNoN/%s/arNoN_%d_%s.txt' %(folder,k,string2))
    d["ar1_{0}".format(k)]  =  np.loadtxt('Pythondata/ar1/%s/ar1_%d_%s.txt' %(folder,k,string2))
#    d["ar2_{0}".format(k)]  =  np.loadtxt('Pythondata/ar2/%s/ar2_%d_%s.txt' %(folder,k,string2))
#    d["ar3_{0}".format(k)]  =  np.loadtxt('Pythondata/ar3/%s/ar3_%d_%s.txt' %(folder,k,string2))
    d["arch_{0}".format(k)]  =  np.loadtxt('Pythondata/arch/%s/arch_%d_%s.txt' %(folder,k,string2))


    
"""Dictionary for tau spectrum"""
tau_d = {}

# for j in range(1,4):
for j in range(1,2):
    tau_d["tau{0}".format(j)] = np.loadtxt('Pythondata/taulists/%s/tau%d_%s.txt' %(folder,j,string2))
    tau_d["tau{0}std".format(j)] = np.loadtxt('Pythondata/taulists/%s/tau%dstd_%s.txt' %(folder,j,string2))

tau_d["tauch"] = np.loadtxt('Pythondata/taulists/%s/tauch_%s.txt' %(folder,string2))
tau_d["tauchstd"]  = np.loadtxt('Pythondata/taulists/%s/tauchstd_%s.txt' %(folder,string2))

"""Aniso: Reproduce tau 2"""
tau2 = psr.tau(3.0,1.5,1.0*mrad,nurange,light)
taugeo = np.sqrt(tau2*taupow4)

"""Reproduce the profile plots"""

#sp = 4      # number of subplots
#
#for k in range(0,len(nurange)):
#    numFig = k/4 + 1
#    print numFig
#    totFig = int(len(nurange)/sp) + 1
#    print "Figure nr: %d/%d" % (numFig,totFig)
#    plt.figure(numFig,figsize=(17,12))
#    subplotcount = (k+1) - sp*(numFig-1)
#    plt.subplot(2,2,subplotcount)
#    plt.subplots_adjust(hspace=.4)             
#    plt.plot(d["arsec"],d["arN_{0}".format(k)]/np.max(d["arNoN_{0}".format(k)]),'g-',alpha = 0.7)
#    plt.plot(d["arsec"],d["ar1_{0}".format(k)]/np.max(d["arNoN_{0}".format(k)]),'r-', linewidth = 2.0,label = r'%s $\tau=%.4f \pm %.4f$' % ('Pre-fold', tau_d["tau1"][k]*bs,tau_d["tau1std"][k]*bs))     
##    plt.plot(d["arsec"],d["ar2_{0}".format(k)]/np.max(d["arNoN_{0}".format(k)]),'b--',linewidth = 2.0, label = r'%s, $\tau=%.4f \pm %.4f$' % ('Post-fold',  tau_d["tau2"][k]*bs,tau_d["tau2std"][k]*bs))     
##    plt.plot(d["arsec"],d["ar3_{0}".format(k)]/np.max(d["arNoN_{0}".format(k)]),'g--',linewidth = 2.0, label = r'%s, $\tau=%.4f \pm %.4f$' % ('EMG',  tau_d["tau3"][k]*bs,tau_d["tau3std"][k]*bs)) 
#    plt.plot(d["arsec"],d["arch_{0}".format(k)]/np.max(d["arNoN_{0}".format(k)]),'c--',linewidth = 2.0, label = r'%s, $\tau=%.4f \pm %.4f$' % ('Pre-fold check' , tau_d["tauch"][k]*bs,tau_d["tauchstd"][k]*bs)) 
#    plt.title(r'%.0f MHz Input: $\tau1=%.3fs$, $\tau2 = %.3fs$, $\tau_{geo} = %.3fs$' % (1000*nurange[k],taupow4[k],tau2[k],taugeo[k]), fontsize=18)    
##    plt.title(r'%.0f MHz, simulated $\tau=%.4f$, ' % (1000*nurange[k],taupow4[k]),fontsize=18) 
#    plt.legend(loc = 'upper right',fontsize=15)
#    plt.xlabel('time (sec)',fontsize=20)
#    plt.xticks(fontsize=18)
#    plt.yticks(fontsize=18)
#    plt.xlim(0,pulseperiod) 
#    filename = 'Aniso_Pulsar_Profiles_%d_%s.png' %(numFig,string2)
#    picpath = "/Users/marisa/Documents/PhD/GitHub_Scattering/Plots/Run100"
#    fileoutput = os.path.join(picpath,filename)
#    plt.savefig(fileoutput, dpi=200)



"""Redo power law fit from obtained tau values"""

def PLM(x,K,k):
    return K*pow(x,-k)
modelpow = Model(PLM)

resultpow = modelpow.fit(tau_d["tau1"],x=nurange,K=0.001,k=4)        
alph1 = resultpow.best_values['k']     
#resultpow2 = modelpow.fit(tau_d["tau2"],x=nurange,K=0.001,k=4)
#alph2 = resultpow2.best_values['k']     
#resultpow3 = modelpow.fit(tau_d["tau3"],x=nurange,K=0.001,k=4)
#alph3= resultpow3.best_values['k']           
resultpowcheck = modelpow.fit(tau_d["tauch"],x=nurange,K=0.001,k=4)
alphch = resultpowcheck.best_values['k']         


"""Reproduce (or change) the ticks marks"""    

ticksMHz = []
for i in range(0,len(nurange),1):
    tMHz = int(round(nurange[i]*1000))
    ticksMHz.append(tMHz)
ticksMHz.append(int(round(1000*nurange[len(nurange)-1])))
ticksMHz = ticksMHz[0:len(ticksMHz):2]

"""Reproduce the tau-spectrum"""

plt.figure(figsize=(14, 9))
plt.errorbar(nurange*1000,tau_d["tau1"]*bs,yerr=tau_d["tau1std"]*bs,fmt='ro',markersize=8, capthick=2,linewidth=1.5, label = r'%s: $\alpha=$ %.3f' % ('Pre-fold' , alph1))
#plt.errorbar(nurange*1000,tau_d["tau2"]*bs,yerr=tau_d["tau2std"]*bs,fmt='b^',markersize=10,capthick=2, linewidth = 1.5,label = r'%s, $\alpha=$ %.3f' % ('Post-fold, long', alph2))
#plt.errorbar(nurange*1000,tau_d["tau3"]*bs,yerr=tau_d["tau3std"]*bs,fmt='g*',markersize=8,capthick=2, linewidth = 1.5,label = r'%s, $\alpha=$ %.3f' % ('EMG', alph3))
plt.errorbar(nurange*1000,tau_d["tauch"]*bs,yerr=tau_d["tauchstd"]*bs,fmt='co',markersize=8,capthick=2, linewidth = 1.5,label = r'%s, $\alpha=$ %.3f' % ('Pre-fold check', alphch))
plt.plot(nurange*1000,resultpow.best_fit*bs,'r--',linewidth = 2.0)
#plt.plot(nurange*1000,resultpow2.best_fit*bs,'b--',linewidth = 2.0)
#plt.plot(nurange*1000,resultpow3.best_fit*bs,'g--',linewidth = 2.0)
plt.plot(nurange*1000,resultpowcheck.best_fit*bs,'c--',linewidth = 2.0)  
#plt.plot(nurange*1000,taupow4,'y*-',markersize=8.0,linewidth=1.5,label = r'$\alpha= 4.00$')    
#plt.plot(nurange*1000,taupow4,'g-',markersize=8.0,linewidth=1.5,label = r'$\tau_{1}$')
#plt.plot(nurange*1000,tau2,'g--',markersize=8.0,linewidth=1.5,label = r'$\tau_{2}$')
plt.plot(nurange*1000,taugeo,'m*-',markersize=8.0,linewidth=1.5,label = r'$\tau_{geo}$')
plt.xlim(nurange[0]*1000-5,nuhigh*1000+5)
#plt.xscale('log')
#plt.yscale('log')
plt.xticks(ticksMHz,ticksMHz,fontsize=26)
plt.yticks(fontsize=26)
plt.xlabel('frequency (MHz)', fontsize=30)
plt.ylabel(r'$\tau$ (sec)',fontsize=30)
plt.legend(loc ='upper right',fontsize=22)


#filenameTau = '%s_TauLin_%s.png' %(folder,string2)
#picpath = "/Users/marisa/Documents/PhD/GitHub_Scattering/Plots/Run100"
#fileoutput = os.path.join(picpath,filenameTau)
#plt.savefig(fileoutput, dpi=200)
#
#
#"""Reproduce the mean and standard deviations from all the iterations (of which the above plots are representitive)"""
#
#alpha1 = np.loadtxt('Pythondata/taulists/%s/alpha1_%s.txt' % (folder,string2))
#alpha2 = np.loadtxt('Pythondata/taulists/%s/alpha2_%s.txt' % (folder,string2))
#alpha3 = np.loadtxt('Pythondata/taulists/%s/alpha3_%s.txt' % (folder,string2))
#alphacheck = np.loadtxt('Pythondata/taulists/%s/alphacheck_%s.txt' %(folder,string2))
#
#mean_alpha1 = np.mean(alpha1)
#mean_alpha2 = np.mean(alpha2)
#mean_alpha3 = np.mean(alpha3)
#mean_alphacheck = np.mean(alphacheck)
#
#std_alpha1 = np.std(alpha1)
#std_alpha2 = np.std(alpha2)
#std_alpha3 = np.std(alpha3)
#std_alphacheck = np.std(alphacheck)
#

""" This section creates a separate dictionary for the RayTracing data"""

"""Dictionary for profiles and model fits"""

"""foldertr = 'RayTrace'
stringtr = 'HF'
pulseperiod = 1.0

draytr = {}

startfreq = 0.04
f_incr = 0.01
endfreq = 0.20
nulow, nuhigh = float(startfreq),float(endfreq)
nurange = np.arange(startfreq,endfreq+f_incr,f_incr)"""


"""Choose arsec.txt for 1.0 sec pulsar and arsec_B.txt for milisecond pulsar."""
"""
#draytr["arsec"]  =  np.loadtxt('Pythondata/arsec/arsec.txt')

for k in range(len(nurange)):    
    draytr["arfit{0}_tr".format(k)]  =  np.loadtxt('Pythondata/%s/arfit%d_tr.txt' %(foldertr,k))

draytr["arN_tr"]  =  np.loadtxt('Pythondata/%s/arN_tr.txt' % foldertr) 
draytr["fluxnorm"] = np.loadtxt('Pythondata/%s/fluxnorm_tr.txt' % foldertr) 

bins = len(draytr["arN_tr"][0])
secs = np.linspace(0,pulseperiod,bins)

fluxn = np.max(draytr["fluxnorm"][0])
tau = np.loadtxt('Pythondata/%s/tau.txt' % foldertr)
obtausec = np.divide(tau,bins)
tauinput = np.loadtxt('Pythondata/RayTrace/taubins.txt')
tauinputsec = np.divide(tauinput,bins)

Reproduce the profile plots
 
sp = 4      # number of subplots

for k in range(0,len(nurange)):
    numFig = k/4 + 1
    print numFig
    totFig = int(len(nurange)/sp) + 1
    print "Figure nr: %d/%d" % (numFig,totFig)
    plt.figure(numFig,figsize=(17,12))
    subplotcount = (k+1) - sp*(numFig-1)
    plt.subplot(2,2,subplotcount)
    plt.subplots_adjust(hspace=.4)             
    plt.plot(secs,draytr["arN_tr"][k]/fluxn,'b')
    plt.plot(secs,draytr["arfit{0}_tr".format(k)]/fluxn,'r--', linewidth = 2.0,label = r'Fitted $\tau: %.2fs$' % obtausec[k])     
    plt.title(r'%.0f MHz, For $\alpha = 4$: $\tau=%.3fs$ ' % (1000*nurange[k],tauinputsec[k]),fontsize=18) 
    plt.legend(loc = 'upper right',fontsize=15)
    plt.xlabel('time (sec)',fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlim(0,pulseperiod)    
    filenametr = 'TruncProfiles_'+str(numFig)+'.png'
    picpath = "/Users/marisa/Documents/PhD/GitHub_Scattering/Plots/RayTrace"
    fileoutput = os.path.join(picpath,filenametr)
    plt.savefig(fileoutput, dpi=200)
"""