# -*- coding: utf-8 -*-
"""
Created on Tue Oct 28 09:49:12 2014

@author: Marisa
"""

import numpy as np
import matplotlib.pyplot as plt
from math import exp,sqrt,pi,erf,log
from scipy.special import iv
#import GPy
#import os

def makeprofile(nbins = 2**9, ncomps = 1, amps = 1, means = 100, sigmas = 10):
    if ncomps == 1:
        npamps = np.array([amps])
        npmeans = np.array([means])
        npsigmas = np.array([sigmas])
    else:    
        npamps = np.array(amps)
        npmeans = np.array(means)
        npsigmas = np.array(sigmas)
   
    profile = np.zeros(nbins)
    x = np.linspace(1,nbins,nbins)
    
    for i in range(ncomps):
#        print npmeans[i]
        profile = profile + \
        npamps[i]*np.exp(-pow(x-npmeans[i],2)/(2*pow(npsigmas[i],2)))
    return x, profile            

def pulsetrain(npulses = 10, bins = np.linspace(1,512,512), profile = np.zeros(512)):
    nbins = np.max(bins)
    train = np.zeros(npulses*int(nbins))

    for i in range(npulses):
        startbin = i*nbins
        train[startbin:startbin + nbins] = profile
    return train
    
#def psrscatter(brfunc, profile):    
#    scattered = np.convolve(profile,brfunc)
#    profint = np.sum(profile)    
#    scint = np.sum(scattered)
#    scattered = scattered / scint * profint
#    bins = profile.shape[0]
#    out = scattered[0:bins]    
#    return out
    
    
def psrscatter(brfunc, profile):    
    scattered = np.convolve(profile,brfunc)
    profint = np.sum(profile)    
    scint = np.sum(scattered)
    scatterednorm = scattered / scint * profint
    bins = profile.shape[0]
    out = scatterednorm[0:bins]    
    return out

def psrscatter_noconserve(brfunc, profile):    
    scattered = np.convolve(profile,brfunc)
#    profint = np.sum(profile)    
#    scint = np.sum(scattered)
#    scattered = scattered / scint * profint
    bins = profile.shape[0]
    out = scattered[0:bins]    
    return out


def psrscatterpostfold(brfunc, profile):    
    scattered = np.convolve(profile,brfunc)
    profint = np.sum(profile)    
    scint = np.sum(scattered)
    scattered = scattered / scint * profint
    length = scattered.shape[0]
    out = scattered[0:length]    
    return out



#def extractpulse(train, pulsesfromend, binsperpulse):
#    start = -pulsesfromend*binsperpulse - 1
#    end = start + binsperpulse    
#    minbin = np.argmin(train[start:end])
#    start2 = start + minbin
#    end2 = start2 + binsperpulse
#    zerobpulse = train[start2:end2]-np.min(train[start2:end2])
#    rectangle = np.min(train[start2:end2])*binsperpulse
#    flux = np.sum(train[start2:end2]) - rectangle
#    return train[start2:end2], zerobpulse, flux

def profilespec(nu,specindex,profile):
    profilefreqspec = []
    for i in range(len(nu)):
        profilefreq = pow(nu[0],specindex)*pow(nu[i],-specindex)*profile
        profilefreqspec.append(profilefreq)
    return profilefreqspec   
    
def sigmaa(kappa,nu):                                       #nu is observing frequency in GHz
        scatteringstrength = kappa*nu**-2                   #kappa is the proportionality constant (radians) and the scattering strength at 1 GHz
        return scatteringstrength    

def tau(D,Ds,kappa,nu,light):                                     #D distance from observer to source in kpc, Ds distance from source to screen in kpc               
    scatteringtime = (D-Ds)*Ds*(sigmaa(kappa,nu))**2/(light*D)    #Tau in seconds
    return scatteringtime




#Create scattered profiles according to different broadening functions respectively and extract a single pulse

### Broadening functions
# 1. Isotropic scattering, isotropic screen

def broadfunc(x,tau):
    broadfunc = (1/tau)*np.exp(-x/tau)
    return broadfunc    

#def broadfuncExt(x,tauE):
#    p1 = np.sqrt(np.pi*tauE/(2*x**3))
#    p22 = []
#    for n in range(1,15,2):    
#        p2a = (n**2*np.pi**2*tauE/(2*x))-1
#        p2b = np.exp(-n**2*np.pi**2*tauE/(4*x))
#        p2 = p2a*p2b
#        p22.append(p2)
#    p22sum = np.sum(p22,axis=0)
#    return p1*p22sum
    
def tauE(D,kappa,nu,light):
    timeconst = 3*D*(sigmaa(kappa,nu))**2/(np.pi**2*light)
    return timeconst

def foldedexpclimb(nexp,period,tau):  #enter in bins (not seconds)
    expfunc2 = []
    tau = float(tau)
    period = float(period)
    periodspace = np.linspace((nexp-1)*period,nexp*period,period)   
    for i in range(nexp):
       expfunc = (1/tau)*np.exp(-(periodspace-i*period)/tau)
       expfunc2.append(expfunc)
       expfunc3 = np.array(expfunc2) 
       expadd = expfunc3.sum(axis=0)
    return expadd 

def foldedexp(nexp,period,tau):  #enter in bins (not seconds)
    expfunc2 = []
    tau = float(tau)
    period = float(period)
    periodspace = np.linspace((nexp-1)*period,nexp*period,period)   
    for i in range(nexp):
       expfunc = (1/tau)*np.exp(-(periodspace-i*period)/tau)
       expfunc2.append(expfunc)
       expfunc3 = np.array(expfunc2) 
       expadd = expfunc3.sum(axis=0)
       expadd = expadd - np.min(expadd)
    return expadd 

# 2. Anisotropic scattering, isotropic screen - Bessell function result
def broadfunc2(x,tau1,tau2):
    broadfunc2 = 1/(np.sqrt(tau1*tau2))*iv(0,x*(1/(2*tau2)-1/(2*tau1)))*np.exp(-x*(1/(2*tau1) + 1/(2*tau2)))
    return broadfunc2  
 
def foldedexp2(nexp,period,tau1,tau2):  #enter in bins (not seconds)
    expfunc2 = []
    tau1 = float(tau1)
    tau2 = float(tau2)
    period = float(period)
    periodspace = np.linspace((nexp-1)*period,nexp*period,period)   
    for i in range(nexp):
       expfunc =  broadfunc2(periodspace-i*period,tau1,tau2)
       expfunc2.append(expfunc)
       expfunc3 = np.array(expfunc2) 
       expadd = expfunc3.sum(axis=0)
       expadd = expadd - np.min(expadd)
    return expadd

def extractpulse(train, pulsesfromend, binsperpulse):
    if pulsesfromend == 0:
        start = 0
        end = binsperpulse
        zerobpulse = train[start:end]-np.min(train[start:end])
        rectangle = np.min(train[start:end])*binsperpulse
        flux = np.sum(train[start:end]) - rectangle
        return train[start:end], zerobpulse, rectangle, flux
    
    else:     
        start = -pulsesfromend*binsperpulse
        end = start + binsperpulse 
        zerobpulse = train[start:end]-np.min(train[start:end])
        rectangle = np.min(train[start:end])*binsperpulse
        flux = np.sum(train[start:end]) - rectangle
        return train[start:end], zerobpulse, rectangle, flux
      
def peaksnr(x, profile, snr):
    bins = profile.shape[0]
    noise = np.random.random(bins)
    peak = np.max(profile)
    out = profile/peak * snr + noise
    return out
    
