# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 12:23:31 2015

@author: marisa
"""

### Importing the EPN profile
### This is done after making small changes in vim to the file EPN123725.txt such that each "\" is removed at the end of the line
### In vim I did this by :%s,\\,,g

import numpy as np
import matplotlib.pyplot as plt
epnprof = np.loadtxt('Pythondata/EPN123725.txt',skiprows = 8)

## Extract the second column which gives the amplitudes of the pulse profile
epnprofy = np.zeros(len(epnprof))

for i in range(len(epnprof)):
    epnprofy[i] = epnprof[i][1]
    
epnprofy = epnprofy/np.max(epnprofy)     
#epnprofy = epnprofy[150:350]
#epnprofy = np.concatenate((np.zeros(80),epnprofy[80:120],np.zeros(80)))
plt.figure()
plt.plot(epnprofy)
plt.show()
