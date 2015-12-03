# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 17:26:06 2015

@author: marisa
"""
"""Sommer net 'n command line copy en paste om die subplts van Fig. 14 in die artikel te maak.
Die profiele wat deur 'n truncated skerm verander is."""

"""Trunc1: Skerm van 400AU by 600AU, Gauss profiel
observednonoise is gestoor by np.savetxt('Pythondata/RayTrace/arNoN_tr.txt', observednonoise)
so behoort maklik te kan redo"""

"""Trunc2: Skerm van 100AU by 100AU met EPN123735 profiel.
observednonoise is hier gestoor by  np.savetxt("Pythondata/EPN123725/Profiles_120_220.txt",observednonoise)
"""
import matplotlib.pyplot as plt
import numpy as np
import os

observednonoise = np.loadtxt('Pythondata/RayTrace/arNoN_tr.txt')
observednonoiseEPN = np.loadtxt('Pythondata/EPN123725/Profiles_120_220.txt')
bins = len(observednonoise[0])
binsEPN = len(observednonoiseEPN[0])

pulseperiod = 1.0
pulseperiodEPN = 1.382  

xsec = np.linspace(0,pulseperiod,bins)
xsecEPN = np.linspace(0,pulseperiodEPN,binsEPN)

startfreq = 0.04
f_incr = 0.01
endfreq = 0.20

startfreqEPN = 0.12
f_incrEPN = 0.02
endfreqEPN = 0.22

nulow, nuhigh = float(startfreq),float(endfreq)
nurange = np.arange(startfreq,endfreq+f_incr,f_incr)

nulowEPN, nuhighEPN = float(startfreqEPN),float(endfreqEPN)
nurangeEPN = np.arange(startfreqEPN,endfreqEPN+f_incrEPN,f_incrEPN)

figuur, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, sharex='col', sharey='row',figsize = (8.2875,  9.05))
#figuur(figsize = (8.2875,  9.05))
#figuur.set_size(8.2875,  9.05)

ax1.set_ylim(0.0,1.0)
ax2.set_ylim(0.0,1.0)
ax3.set_ylim(0.0,1.0)
ax4.set_ylim(0.0,1.0)
ax5.set_ylim(0.0,1.0)
ax6.set_ylim(0.0,1.0)

xl = 0.00
xr = 1.00
xll = 0.50
xrr = 0.90

ax1.set_xlim(xl,xr)
ax2.set_xlim(xll,xrr)
ax3.set_xlim(xl,xr)
ax4.set_xlim(xll,xrr)
ax5.set_xlim(xl,xr)
ax6.set_xlim(xll,xrr)

"""Vir die truncated screen, plot ek elke tweede gestoorde frek"""
ax1.plot(xsec,observednonoise[0]/np.max(observednonoise[0]),'b',linewidth = 4.0)
ax2.plot(xsecEPN,observednonoiseEPN[0]/np.max(observednonoiseEPN[0]),'m',linewidth = 4.0)
ax3.plot(xsec,observednonoise[4]/np.max(observednonoise[4]),'b',linewidth = 4.0)
ax4.plot(xsecEPN,observednonoiseEPN[1]/np.max(observednonoiseEPN[1]),'m',linewidth = 4.0)
ax5.plot(xsec,observednonoise[8]/np.max(observednonoise[8]),'b',linewidth = 4.0)
ax6.plot(xsecEPN,observednonoiseEPN[2]/np.max(observednonoiseEPN[2]),'m',linewidth = 4.0)


ax1.set_title('%.2f MHz'% (nurange[0]*1000), fontsize=25)
ax2.set_title('%.2f MHz'% (nurangeEPN[0]*1000), fontsize=25)
ax3.set_title('%.2f MHz'% (nurange[4]*1000), fontsize=25)
ax4.set_title('%.2f MHz'% (nurangeEPN[1]*1000), fontsize=25)
ax5.set_title('%.2f MHz'% (nurange[8]*1000), fontsize=25)
ax6.set_title('%.2f MHz'% (nurangeEPN[2]*1000), fontsize=25)

"""Vir die truncated EPN profiel, plot ek elke frek. (Die kode het in incr 20 MHz gehardloop)"""
#ax1.plot(xsec,observednonoise[0]/np.max(observednonoise[0]),'m',linewidth = 4.0)
#ax2.plot(xsec,observednonoise[1]/np.max(observednonoise[1]),'m',linewidth = 4.0)
#ax3.plot(xsec,observednonoise[2]/np.max(observednonoise[2]),'m',linewidth = 4.0)
#ax4.plot(xsec,observednonoise[3]/np.max(observednonoise[3]),'m',linewidth = 4.0)
#ax5.plot(xsec,observednonoise[4]/np.max(observednonoise[4]),'m',linewidth = 4.0)
#ax6.plot(xsec,observednonoise[5]/np.max(observednonoise[5]),'m',linewidth = 4.0)
#
#
#ax1.set_title('%.2f MHz'% (nurange[0]*1000), fontsize=25)
#ax2.set_title('%.2f MHz'% (nurange[1]*1000), fontsize=25)
#ax3.set_title('%.2f MHz'% (nurange[2]*1000), fontsize=25)
#ax4.set_title('%.2f MHz'% (nurange[3]*1000), fontsize=25)
#ax5.set_title('%.2f MHz'% (nurange[4]*1000), fontsize=25)
#ax6.set_title('%.2f MHz'% (nurange[5]*1000), fontsize=25)


ax1.tick_params(axis='both', which='major', labelsize=16)
ax2.tick_params(axis='both', which='major', labelsize=16)
ax3.tick_params(axis='both', which='major', labelsize=16)
ax5.tick_params(axis='both', which='major', labelsize=16)
ax6.tick_params(axis='both', which='major', labelsize=16)

#plt.text(0.45, -0.3, 'time (sec)', ha='center', va='center',fontsize=25)
plt.text(0.45, -0.3, 'time (sec)', ha='center', va='center',fontsize=25)
#plt.setp(ax5, xticks=[0.5, 0.6, 0.7, 0.8, 0.9])
plt.setp(ax6, xticks=[0.5, 0.6, 0.7, 0.8, 0.9])
#plt.setp(ax5, xticks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
#plt.setp(ax6, xticks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

#filename = 'EPNProfiles.png'
#picpath = "/Users/marisa/Documents/PhD/GitHub_Scattering/Pythondata/EPN123725"
#fileoutput = os.path.join(picpath,filename)
#plt.savefig(fileoutput, dpi=200)

filename = 'Truncprofiles_Both.png'
#picpath = "/Users/marisa/Documents/PhD/GitHub_Scattering/Pythondata/RayTrace"
picpath = "/Users/marisa/Dropbox/Aris/TexOutputs/ScatteringPaperTexAndImages/Spyderplots/RayTrace"
fileoutput = os.path.join(picpath,filename)
plt.savefig(fileoutput, dpi=200)


"""Figure size I'm using is:"""
"""fig = plt.gcf()
size = fig.get_size_inches()
size
Out[362]: array([ 8.2875,  9.05  ])"""

