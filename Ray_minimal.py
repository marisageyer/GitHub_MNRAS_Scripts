import sys
import matplotlib.pyplot as plt
from matplotlib.pyplot import ion, draw
from scipy.special import erf
import numpy as np
#import argparse
import time
from mpl_toolkits.mplot3d.axes3d import Axes3D

##### Set up functions that produce photons on scattering screen and computes the likelihood of reaching the detector#       1. photons - for generating photons on screen centered on line of sight (impacts = 0)
#       1. photons_on_screen - generate photons on a finite screen with screensize_x by screensize_y 
#                             - can be off-centre to the line of sight by choosing, impactx and/or impacty non-zero
#       2. photons_that_hit - calculates the prob of a given photon on the chosen screen hitting the observer

def photons_on_screen(impactx,impacty,screensize_x, screensize_y,res):
    # N number of photons    
    # impactx and y are the coordinates of the intersection of the
    # line of sight to the sources with the screen in AU w.r.t the
    # centre of the screen
    # screensize_x,y are the half-diameter sizes of the screen
    lowx = impactx - screensize_x  
    highx = impactx + screensize_x
    lowy = impacty - screensize_y  
    highy = impacty + screensize_y
#    xx_scr, yy_scr = np.random.uniform(low=lowx, high=highx, size=size), np.random.uniform(low=lowy, high=highy,size=size)    
    xx_scr, yy_scr = np.arange(lowx,highx,resAU), np.arange(lowy,highy,resAU)
#    X_scr, Y_scr = np.meshgrid(xx_scr,yy_scr)
    return xx_scr, yy_scr
    
#def photons_on_screen_random(impactx, impacty,screensize_x, screensize_y, size):
#    lowx = impactx - screensize_x  
#    highx = impactx + screensize_x
#    lowy = impacty - screensize_y  
#    highy = impacty + screensize_y
#    xx_scr, yy_scr = np.random.uniform(low=lowx, high=highx, size=size), np.random.uniform(low=lowy, high=highy,size=size)
#    xx_lin, yy_lin = np.linspace(lowx,highx,size), np.linspace(lowy,highy,size)
#    xx_res, yy_res = xx_lin[2]-xx_lin[1], yy_lin[2]-yy_lin[1]
#    return xx_scr, yy_scr, xx_res, yy_res

def photons_that_hit(xx, yy, Ds,D,sigma_a1_1GHz,sigma_a2_1GHz, freq,res):
    # xx, yy are the (x,y) coordinates of the photons on the scattering screen as produced by funcs photons or photons_on_screen
    # Ds is the straight-line distance from the source to the screen in kpc
    # D is the straight-line distance from the source to the observer in kpc
    # sigma_a_1GHz is the scattering strength at 1 GHz in mas, typically chosen to be 3mas from CordesLazio2001   
    # Several necessary conversion factors are also included below  
    # theta_hit and phi_hit: corresponding (theta,phi) of the (x,y) coordinates along which the photons on the screen need to travel to hit the detector  
    # probX,probY, eps:  prob of photon travelling with the right angle in the x,y-direction to hit detector, eps: stepsize in angle (rad) over which the probs are calculated     
    # time_x,time_y,time_r: time delays associated with the extra pathlength along the x,y-direction and combined delay, time_r  
    sigma_a1 = sigma_a1_1GHz*freq**-2
    sigma_a2 = sigma_a2_1GHz*freq**-2
    mas2rad = 4.85e-9
    sigma_a1_rad = sigma_a1*mas2rad
    sigma_a2_rad = sigma_a2*mas2rad
    light = 9.71561189e-12                      # in kpc/s     
    kpc2AU = 206264806         
    theta_hit = np.arctan(xx/((D-Ds)*kpc2AU)) 
    phi_hit = np.arctan(yy/((D-Ds)*kpc2AU))
    thres = theta_hit[2] - theta_hit[1]
    phires = phi_hit[2]- phi_hit[1]
    sigma_th_rad = (Ds/D)*sigma_a1_rad
    sigma_phi_rad = (Ds/D)*sigma_a2_rad                                
    gaux = (1.0/(sigma_th_rad*np.sqrt(2*np.pi)))*np.exp(-np.power(theta_hit,2)/(2.0*np.power(sigma_th_rad,2)))
    gauy =  (1.0/(sigma_phi_rad*np.sqrt(2*np.pi)))*np.exp(-np.power(phi_hit,2)/(2.0*np.power(sigma_phi_rad,2)))   
    probX = gaux*thres
    probY = gauy*phires
    pxytable = []
    for i in range(len(probX)):
        pxy = probX[i]*probY
        pxytable.append(pxy)
        pXY = np.array([pxytable])
    pXY = pXY.reshape(len(probX)*len(probY))
    dist_x = (np.sqrt(np.power(Ds,2) + np.power(xx/kpc2AU,2)) + 
         np.sqrt(np.power((D-Ds),2) + np.power(xx/kpc2AU,2)))
    time_x = D/light - dist_x/light    
    dist_y = (np.sqrt(np.power(Ds,2)+ np.power(yy/kpc2AU,2)) +
        np.sqrt(np.power((D-Ds),2)+ np.power(yy/kpc2AU,2)))
    time_y = D/light - dist_y/light
    delaytable = []
    for i in range(len(time_x)):
        txy = np.sqrt(np.power(time_x[i],2)+ np.power(time_y,2))
        delaytable.append(txy)
        delay = np.array([delaytable])
    delay = delay.reshape(len(time_x)*len(time_y))             
    return delay,probX,probY,pXY


def photons_infscreen(sizex,sizey,Ds,D,sigma_a1_1GHz,sigma_a2_1GHz,freq,res):
    sigma_a1 = sigma_a1_1GHz*freq**-2    
    sigma_a2 = sigma_a2_1GHz*freq**-2        
    mas2rad = 4.85e-9
    sigma_a1_rad = sigma_a1*mas2rad
    sigma_a2_rad = sigma_a2*mas2rad      
    kpc2AU = 206264806
    xinf = np.arange(-sizex,sizex,res)
    yinf = np.arange(-sizey,sizey,res)    
    theta_hit = np.arctan(xinf/((D-Ds)*kpc2AU)) # spread in angles that hit the detector
    phi_hit = np.arctan(yinf/((D-Ds)*kpc2AU))  
    infres = theta_hit[2] - theta_hit[1]
    sigma_th_rad = (Ds/D)*sigma_a1_rad
    sigma_phi_rad = (Ds/D)*sigma_a2_rad
    gauxinf = (1.0/(sigma_th_rad*np.sqrt(2*np.pi)))*np.exp(-np.power(theta_hit,2)/(2.0*np.power(sigma_th_rad,2)))
    gauyinf = (1.0/(sigma_phi_rad*np.sqrt(2*np.pi)))*np.exp(-np.power(phi_hit,2)/(2.0*np.power(sigma_phi_rad,2)))    
    px = gauxinf*infres
    py = gauyinf*infres                          
    infpxy = np.sum(px)*np.sum(py)
    #This will change when the sigmas differ           
    return gauxinf, gauyinf, infpxy



## Code execution starts here

starttime = time.time()
trdatabinnedfreq = []


## Choose a time resolution and compute the relevant spatial resolution

light = 9.71561189e-12                      # in kpc/s     
kpc2AU = 206264806
mas2rad = 4.85e-9

## Choose scattering setup parameters
   
#F = 0.15                # in GHz
DD = 3.0                # in kpc
Dss = 1.5               # in kpc
sigma1 = 3.0            # in mas; in the article I have used sigma1 = sigma2 = 3.0 for isotropic scattering
sigma2 = 3.0            # and sigma1 = 3.0, sigma2 = 1.0 for anisotropic scattering.


impactx = 0
impacty = 0
ssizex =  50        # The sizes here are half-diameter sizes of the rectangle. 
ssizey = 100         # So overall width, height is 2*ssizex, 2*ssizey      


#impactx = 0             # These are the size and impact
#impacty = 0             # parameters I choose to use for
#ssizex = 100            # the EPN profile of B1237+25
#ssizey = 100
#DD = 0.85
#Dss = 0.85/2.0
#sigma1 = 3.0
#sigma2 = 3.0

print ("D: %.2f kpc" % DD)
print ("Ds: %.2f kpc" % Dss)
print ("sigma @ 1GHz: %d mas" % sigma1) 


## Determine the spatial resolution that will produce the desired temporal resolution
global resAU

timeres = 0.5*1e-6
#timeres = 0.5*1e-5
Dsprime = Dss*(1 - Dss/DD)
resAU = (DD - Dss)*kpc2AU*np.sqrt(2*light*timeres/Dsprime)*(Dss/DD)



#Observing frequencies
startfreq = 0.04
f_incr = 0.01
endfreq = 0.44
nulow, nuhigh = float(startfreq),float(endfreq)
nurange = np.arange(startfreq,endfreq+f_incr,f_incr)


##Infinite screen reference
# Calculate size that approximates an infinite screen
# Assume 5*sigma is a valid approx. for having received all photos (for flux conservation)


## Calculate chosen broadening function as a function of frequency

probfreq = []
binvalfreq = []

#calculate the maximum time of the edge of the screen at the lowest frequency

xmax = ssizex + abs(impactx)
ymax = ssizey + abs(impacty)

dmax_x = (np.sqrt(np.power(Dss,2) + np.power(float(xmax)/kpc2AU,2)) + 
         np.sqrt(np.power((DD-Dss),2) + np.power(float(xmax)/kpc2AU,2)))
tmax_x = dmax_x/light - DD/light   
dmax_y = (np.sqrt(np.power(Dss,2)+ np.power(float(ymax)/kpc2AU,2)) +
    np.sqrt(np.power((DD-Dss),2)+ np.power(float(ymax)/kpc2AU,2)))
tmax_y = dmax_y/light - DD/light
tmax = np.sqrt(np.power(tmax_x,2)+ np.power(tmax_y,2))              

print ("tmax: %f" %tmax)

#sys.exit()


#plt.figure()
for i in range(len(nurange)):   
    F = startfreq + f_incr*i
    print("")
    print ("Freq: %.2f" %F)
    sigref = 5
    infhalfscreenx = (DD-Dss)*np.tan(sigref*sigma1*np.power(F,-2)*mas2rad*Dss/DD)*kpc2AU     #This gives the half-screensize of the chosen reference infinite screen (as determined by the 5sigma size)
    infhalfscreeny = (DD-Dss)*np.tan(sigref*sigma2*np.power(F,-2)*mas2rad*Dss/DD)*kpc2AU    
    print ("Ref. x-size of inf screen: %.2f AU" %infhalfscreenx)  
    print ("Ref. y-size of inf screen: %.2f AU" %infhalfscreeny) 


    xxtr, yytr = photons_on_screen(impactx,impacty,ssizex,ssizey,resAU)
#    xxtr, yytr = photons_on_screen(numbdilute,impactx,impacty,infhalfscreen,infhalfscreen)
    trtime,trpx,trpy,tpxy = photons_that_hit(xxtr,yytr,Dss,DD,sigma1,sigma2,F,resAU)
    gauinfx, gauinfy, infprob = photons_infscreen(infhalfscreenx,infhalfscreeny,Dss,DD,sigma1,sigma2,F,resAU)
  
#    plt.plot(thinf[0:len(thinf):1000], gaus[0:len(thinf):1000],'m.')
#    plt.plot(th[0:len(th):1000],trpx[0:len(th):1000],'b*')
#    plt.plot(phi[0:len(phi):1000],trpy[0:len(phi):1000],'y.')      

    print ("sum of tr_prob x: %.2f,sum of tr_prob y: %.2f " % (np.sum(trpx), np.sum(trpy)))
    print ("Combined probability: %.2f" %np.sum(tpxy))
    print ("Check inf prob: %.2f" %infprob)

    trtimedata = np.column_stack((trtime,tpxy))
    trtimesort = trtimedata[trtimedata[:,0].argsort()]
    
    trproborder = trtimesort[:,1]  #photon probabilities counts in chronological order
    trtimeorder = trtimesort[:,0]

#    binstimeres = 0.5*1e-3
    binstimeres = 1e-3
#    binstimeres = 1e-4
    binvals = np.arange(0.,tmax,binstimeres) 
          
    hist, bined = np.histogram(trtimeorder,binvals)
    
    histsum = []
    
    for k in range(len(hist)):
        histsums  = np.sum(hist[0:k])
        histsum.append(histsums)
    
    probbinned = []
    for k in range(len(hist)-1):
        probarange = np.sum(trproborder[histsum[k]:histsum[k+1]])
        probbinned.append(probarange)
    
    binvals2 = binvals[0:len(probbinned)]   
    print ("Length binval: %.0f" %len(binvals2))
    print("")
 

    binvalfreq.append(binvals2)
    probfreq.append(probbinned)

probfreq = np.array(probfreq)

print("ressample/binres: %.2e" %(timeres/binstimeres))

#
#photonsinterfreq = []

mm = 3. #factor by which to up the sampling rate.
photonsinterpp = []

#plt.figure()
#for k in range(len(probfreq)):
#    interptime = np.linspace(binvalfreq[k][0],binvalfreq[k][len(binvalfreq[k])-1],mm*len(binvalfreq[k]))
#    binstimeresinterp = interptime[1]-interptime[0]
#    photonsinterp = np.interp(interptime,binvalfreq[k],probfreq[k])
#photonsinterfreq.append(photonsinterp)
#plt.plot(binvals2[k],photonsfreq[k], linewidth= 2.0, label = '%.2f GHz' %F)
#    plt.plot(binvalfreq[k],probfreq[k],linewidth= 2.0, label = '%.0f MHz' %(nurange[k]*1000))  
#    plt.figure()
#    plt.plot(interptime,photonsinterp,'go',alpha=0.5)
#    plt.plot(binvalfreq[k],probfreq[k],linewidth= 2.0, label = 'Impactx = %.0f,Impacty = %.0f AU' %(impactx,impacty))   
#plt.plot(bintimeres,photonsinterp, linewidth= 2.0, label = '%.2f GHz' %F)   
#plt.plot(trbintime,trdatabinned, linewidth= 2.0, label = 'impactx: %d AU, impacty: %d AU' % (impactx,impacty)) 
#plt.plot(trbintime,trdatabinned, linewidth= 2.0, label ='%d AU' % ssizex) 
#plt.plot(bintime, databinned, 'r--') 
#plt.xlabel("time (sec)",fontsize=18)
#plt.ylabel("photon probability",fontsize=18)
#plt.legend(loc = 'best', fontsize=14)
#plt.xticks(fontsize=16)
#plt.yticks(fontsize=16)
#plt.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))
#    photonsinterpp.append(photonsinterp)
    
#for no numbers on an axis use, alpha = 0.0


fluxtr = []
for i in range(len(probfreq)):
    fltr = np.sum(probfreq[i])
    fluxtr.append(fltr)
    
fluxtr = np.array(fluxtr)

#samp = 2
#timeressamp = binstimeres*samp
#
#probf = []
#for i in range(len(probfreq)):
#    prow = probfreq[i]
#    psamp = prow[0:len(prow):samp]
#    probf.append(psamp)
#
#probf = np.array(probf)
    

#photonsinterfreq = np.array(photonsinterfreq)


print ("------ %s seconds ------" %(time.time() - starttime))



