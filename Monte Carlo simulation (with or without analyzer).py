# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 12:08:19 2022

@author: apm41

delete analysis of triplet state kexc from ed10, add secondary delay into function
"""

import numpy as np
from random import random
import matplotlib.pyplot as plt
import pandas as pd
from math import sqrt
from scipy.stats import norm
from scipy.optimize import curve_fit
import time 
# In[] Functions:

'''
1. Rotational Brownian motion
2. Rate constant
3. Exponential decay generator
4. Jumping between states
    4.1 S0
    4.2 S1
    4.3 T1
    4.4 T2 
5. OADFA simulation
6. Exponential fitting
'''
################### rotational brownian motion #############################
def r_brownian(a0, n, dt, delta, counts):     
    # Generate random numbers obeyed normal distribution. 
    # a0: initial angle. n: number of steps. dt: bin size (time) of each step. delta: rotational rate factor. counts: number of molecules
    # rvs(loc=0, scale=1, size=1, random_state=None) #loc=mean, scale=stdev, size=number of data)
    r = norm.rvs(size = (n,)+ (counts,), scale=delta*sqrt(dt)) 
    
    # Create an empty array for output storage
    result = np.empty(r.shape)
    
    # This computes the Brownian motion by forming the cumulative sum of the random samples. 
    np.cumsum(r, axis=0, out=result)
    
    # Add the initial angle.
    np.add(result, a0, out = result) 
    
    # Create an array for initial angle, and put two arrays together to get the final result
    if type(a0) is float:
        x = np.full((1,counts),a0)    # a new array with every element equals to a0
    elif a0.shape == (1, counts):
        x = np.asarray(a0)
    
    result = np.vstack([x,result])

    return result

########## Method to calculate intensity-dependent excitation rates, including saturation #############3
def kexc(Int,sig,wavel,ISat):
    conv = wavel/(h*c)
    return sig*Int*conv/(1+Int/ISat)

########## Exponential decay generator ##########
def gen( a ):
    y=random()
    return( -np.log( y ) / a )
def dist_func( x, a ):
    return(np.exp( -a * x) )

########## Jumping between states ###########
def atS0(t,IPrim, angle):
    pho_sele = np.cos(angle)**2 
    k01 = kexc(IPrim*pho_sele,sigS0,waveP,Isat)
    
    if random() >=  dist_func( t, k01 ):
        return 'S1'    # goes to S1 at time t
    else:
        return 'S0'  # stay in S0 at time t

def atS1(t):
     rand1 = random()
     if rand1 <= QYF:
         if random() >=  dist_func( t, (kF+kISC+kISC2+kIC) ):
             return 'Fl-S0'    # goes to S0 with fluorescence emitted at time t
         else:
             return 'S1'    # stay in S1 at time t
     elif QYF < rand1 <= (QYF+QYT):
          if random() >=  dist_func( t, (kF+kISC+kISC2+kIC) ):
              return 'ISC-T1'    # goes to T1 at time t
          else:
              return 'S1'    # stay in S1 at time t
     elif (QYF+QYT) < rand1 <= (QYF+QYT+QYT2):
          if random() >=  dist_func( t, (kF+kISC+kISC2+kIC) ):
              return 'ISC-T2'    # goes to T2 at time t
          else:
              return 'S1'    # stay in S1 at time t
     else:
         if random() >=  dist_func( t, (kF+kISC+kISC2+kIC) ):
             return 'IC-S0'
         else:
             return 'S1'    # stay in S1 at time t
 
def atT1(t,Isec, angle, polar):
    if polar == 'para':
        pho_sele = np.cos(angle)**2
    elif polar == 'perp':
        pho_sele = np.sin(angle)**2
    elif polar == 'non':
        pho_sele = 1

    kexc_sec = kexc(Isec*pho_sele,sigT1,waveS,Isat)
    QYTn1 =  kexc_sec/(kexc_sec+kT)
    rand1 = random() # goes to S0 or Tn
    rand2 = random() # move or stay in the same state
    if rand1 <= QYTn1:
        if rand2 >=  dist_func( t, kexc_sec + kT ) and random() <= QYrev1:
            return 'S1', kexc_sec    # goes to Tn1 at time t
        else:
            return 'T1', 0  # stay in T1 at time t
    elif rand1 > QYTn1:
        if rand2 >=  dist_func( t, kexc_sec + kT ):
            return 'S0', 0    # goes to S1 at time t
        else:
            return 'T1', 0  # stay in Tn1 at time t


def atT2(t,Isec, angle, polar):
    if polar == 'para':
        pho_sele = np.cos(angle)**2 
    elif polar == 'perp':
        pho_sele = np.sin(angle)**2
    elif polar == 'non':
        pho_sele = 1

    kexc_sec = kexc(Isec*pho_sele,sigT1,waveS,Isat)
    QYTn2 =  kexc_sec/(kexc_sec+kT2)
    rand1 = random() # goes to S0 or Tn
    rand2 = random() # move or stay in the same state
    if rand1 <= QYTn2:
        if rand2 >=  dist_func( t, kexc_sec + kT2 ) and random() <= QYrev2:
            return 'S1'    # goes to Tn2 at time t
        else:
            return 'T2'  # stay in T2 at time t
    elif rand1 > QYTn2:
        if rand2 >=  dist_func( t, kexc_sec + kT2 ):
            return 'S0'    # goes to S1 at time t
        else:
            return 'T2'  # stay in Tn2 at time t

#####OADFA simulation function######
def OADFA_Simu(t_series, Event_num, angle, Ipara, Iperp, counter):
    start_t2 = time.time() # Stopwatch start
    
    
    state = np.empty([len(t_series), Event_num], dtype=object)
    state[:] = str('S0')
    para_event = 0
    ####  Ipara ####
    for j in range(Event_num):
        for i in range(len(t_series)-1):
            if state[i, j] == 'S0':
                temp = 'S0'
                if t_series[i] < pPulseW:
                    temp = atS0(t_series[i+1]-t_series[i],IPpulsed, angle[i, j]) 
                state[i+1, j] = str(temp)
            elif state[i, j] =='S1':
                temp = atS1(t_series[i+1]-t_series[i])
                if temp =='Fl-S0':
                    state[i+1, j] = str('S0')
                    Ipara[i] += 1
                    # Ipara[i] += np.cos(angle[i, j])**2
                    # Iperp[i] += (np.sin(angle[i, j])**2 )#*(np.sin(y[i, j])**2 )
                elif temp =='ISC-T1':
                    state[i+1, j] = str('T1')
                elif temp =='ISC-T2':
                    state[i+1, j] = str('T2')   
                elif temp =='IC-S0':
                    state[i+1, j] = str('S0')  
                else:
                    state[i+1, j] = str(temp)
            elif state[i, j] =='T1':
                if t_series[i+1] < delay:
                    temp = atT1(t_series[i+1]-t_series[i],0, angle[i, j], 'para') 
                elif t_series[i+1] >= delay:
                    temp = atT1(t_series[i+1]-t_series[i],ISMax, angle[i, j], 'para') 
                state[i+1, j] = str(temp[0])
                para_event += 1
            elif state[i, j] =='T2':
                if t_series[i+1] < delay:
                    temp = atT2(t_series[i+1]-t_series[i],0, angle[i, j], 'para') 
                elif t_series[i+1] >= delay:
                    temp = atT2(t_series[i+1]-t_series[i],ISMax, angle[i, j], 'para')  
                state[i+1, j] = str(temp)
    del state
    ####  Iperp ####
    state = np.empty([len(t_series), Event_num], dtype=object)
    state[:] = str('S0')
    perp_event = 0
    for j in range(Event_num):
        for i in range(len(t_series)-1):
            if state[i, j] == 'S0':
                temp = 'S0'
                if t_series[i] < pPulseW:
                    temp = atS0(t_series[i+1]-t_series[i],IPpulsed, angle[i, j]) 
                state[i+1, j] = str(temp)
            elif state[i, j] =='S1':
                temp = atS1(t_series[i+1]-t_series[i])
                if temp =='Fl-S0':
                    state[i+1, j] = str('S0')
                    Iperp[i] += 1
                    # Ipara[i] += np.cos(angle[i, j])**2
                    # Iperp[i] += (np.sin(angle[i, j])**2 )#*(np.sin(y[i, j])**2 )
                elif temp =='ISC-T1':
                    state[i+1, j] = str('T1')
                elif temp =='ISC-T2':
                    state[i+1, j] = str('T2')   
                elif temp =='IC-S0':
                    state[i+1, j] = str('S0')  
                else:
                    state[i+1, j] = str(temp)
            elif state[i, j] =='T1':
                if t_series[i+1] < delay:
                    temp = atT1(t_series[i+1]-t_series[i],0, angle[i, j], 'perp') 
                elif t_series[i+1] >= delay:
                    temp = atT1(t_series[i+1]-t_series[i],ISMax, angle[i, j], 'perp') 
                state[i+1, j] = str(temp[0])
                perp_event +=1
            elif state[i, j] =='T2':
                if t_series[i+1] < delay:
                    temp = atT2(t_series[i+1]-t_series[i],0, angle[i, j], 'perp') 
                elif t_series[i+1] >= delay:
                    temp = atT2(t_series[i+1]-t_series[i],ISMax, angle[i, j], 'perp')  
                state[i+1, j] = str(temp)
    del state
        #### Primary only FA simulation #####
    
    
    # calculate and show the number of molecule involved
    counter += 1
    # print('number of molecules =', counter*Event_num) 
    
    end_t2 = time.time() # Stopwatch stop
    
    print('processing time(OADFA)-', counter,'=', 
          int((end_t2-start_t2)//3600), ':', int((end_t2-start_t2)%3600//60), ':', int((end_t2-start_t2)%60)) # print stopwatch result
    
    del start_t2, end_t2
    return(Ipara, Iperp, counter)


def OADFA_Analyzer_Simu(t_series, Event_num, angle, Ipara, Iperp, counter):
    start_t2 = time.time() # Stopwatch start
    start_t2 = time.time() # Stopwatch start
    
    
    state = np.empty([len(t_series), Event_num], dtype=object)
    state[:] = str('S0')
    para_event = 0
    ####  Ipara ####
    for j in range(Event_num):
        for i in range(len(t_series)-1):
            if state[i, j] == 'S0':
                temp = 'S0'
                if t_series[i] < pPulseW:
                    temp = atS0(t_series[i+1]-t_series[i],IPpulsed, angle[i, j]) 
                state[i+1, j] = str(temp)
            elif state[i, j] =='S1':
                temp = atS1(t_series[i+1]-t_series[i])
                if temp =='Fl-S0':
                    state[i+1, j] = str('S0')
                    # Ipara[i] += 1
                    Ipara[i] += np.cos(angle[i, j])**2
                    Iperp[i] += (np.sin(angle[i, j])**2 )#*(np.sin(y[i, j])**2 )
                elif temp =='ISC-T1':
                    state[i+1, j] = str('T1')
                elif temp =='ISC-T2':
                    state[i+1, j] = str('T2')   
                elif temp =='IC-S0':
                    state[i+1, j] = str('S0')  
                else:
                    state[i+1, j] = str(temp)
            elif state[i, j] =='T1':
                if t_series[i+1] < delay:
                    temp = atT1(t_series[i+1]-t_series[i],0, angle[i, j], 'para') 
                elif t_series[i+1] >= delay:
                    temp = atT1(t_series[i+1]-t_series[i],ISMax, angle[i, j], 'para') 
                state[i+1, j] = str(temp[0])
                para_event += 1
            elif state[i, j] =='T2':
                if t_series[i+1] < delay:
                    temp = atT2(t_series[i+1]-t_series[i],0, angle[i, j], 'para') 
                elif t_series[i+1] >= delay:
                    temp = atT2(t_series[i+1]-t_series[i],ISMax, angle[i, j], 'para')  
                state[i+1, j] = str(temp)
    del state
    ####  Iperp ####
    state = np.empty([len(t_series), Event_num], dtype=object)
    state[:] = str('S0')
    perp_event = 0
    for j in range(Event_num):
        for i in range(len(t_series)-1):
            if state[i, j] == 'S0':
                temp = 'S0'
                if t_series[i] < pPulseW:
                    temp = atS0(t_series[i+1]-t_series[i],IPpulsed, angle[i, j]) 
                state[i+1, j] = str(temp)
            elif state[i, j] =='S1':
                temp = atS1(t_series[i+1]-t_series[i])
                if temp =='Fl-S0':
                    state[i+1, j] = str('S0')
                    # Iperp[i] += 1
                    Ipara[i] += np.cos(angle[i, j])**2
                    Iperp[i] += (np.sin(angle[i, j])**2 )#*(np.sin(y[i, j])**2 )
                elif temp =='ISC-T1':
                    state[i+1, j] = str('T1')
                elif temp =='ISC-T2':
                    state[i+1, j] = str('T2')   
                elif temp =='IC-S0':
                    state[i+1, j] = str('S0')  
                else:
                    state[i+1, j] = str(temp)
            elif state[i, j] =='T1':
                if t_series[i+1] < delay:
                    temp = atT1(t_series[i+1]-t_series[i],0, angle[i, j], 'perp') 
                elif t_series[i+1] >= delay:
                    temp = atT1(t_series[i+1]-t_series[i],ISMax, angle[i, j], 'perp') 
                state[i+1, j] = str(temp[0])
                perp_event +=1
            elif state[i, j] =='T2':
                if t_series[i+1] < delay:
                    temp = atT2(t_series[i+1]-t_series[i],0, angle[i, j], 'perp') 
                elif t_series[i+1] >= delay:
                    temp = atT2(t_series[i+1]-t_series[i],ISMax, angle[i, j], 'perp')  
                state[i+1, j] = str(temp)
    del state
        #### Primary only FA simulation #####
    
    
    # calculate and show the number of molecule involved
    counter += 1
    # print('number of molecules =', counter*Event_num) 
    
    end_t2 = time.time() # Stopwatch stop
    
    print('processing time(OADFA)-', counter,'=', 
          int((end_t2-start_t2)//3600), ':', int((end_t2-start_t2)%3600//60), ':', int((end_t2-start_t2)%60)) # print stopwatch result
    
    del start_t2, end_t2
    return(Ipara, Iperp, counter)

########## Exponential fitting #############
def expo(x, A, k, y0):
    return A * np.exp(-x/k) + y0 # exponatial fitting
def biexpo(x, A1, k1, A2, k2, y0):
    return A1 * np.exp(-x/k1) + A2 * np.exp(-x/k2) + y0 # exponatial fitting

# In[] Parameter Setting

###### Define parameters ... these first few are fundamental constants, and (obviously) shouldnt change ####
global h,c,nA
h = 6.626*10**-34
c = 2.998*10**8
nA = 6.022 * 10**23
###### molecular parameters for rate matrices - Adjust these as needed for your dye #####
global kISC,kISC2,tF,tT,kf,kIC,kT,kT2,QYrev1,QYrev2,sigS0,sigT1,sigD2

QYF = 0.1 ### Fluorescence QY
tF = 2.7 *10**-9 
kF = QYF/tF 
QYT = 0.5 ## Triplet QY
QYT2 = 0.00 ## Triplet QY
kISC = QYT/tF
kISC2 = QYT2/tF
tT = 6.5 * 10**-6  *30
tT2 = 21.5 * 10**-6
epS0 = 2.6 * 10**5 ### ground state extinction coefficient

sigT1 = 0.37*10**-18/0.2
sigT2 = 0.13*10**-18/0.2
######## Adjust parameters above this line.... #########

kIC = (1 - kF * tF - QYT - QYT2)/tF
kT = 1/tT        ## Natural triplet state decay rate
kT2 = 1/tT2      ## Natural triplet state decay rate2
QYrev1 = 0.9 #0.07 for MC540
QYrev2 = 0.0 #QYrev1/20
#epD2 = 0. #1.3 * 10**5
sigS0 = epS0/nA *1000*np.log(10)

epT1 = sigT1/np.log(10)/1000*nA ### dark state extinction coefficient
epT2 = sigT2/np.log(10)/1000*nA ### dark state extinction coefficient

##### laser parameters common to pulsed and cw excitation schemes - Adjust as needed
global waveP, waveS, period,Isat
waveP = 532 * 10**-9 ## wavelength of primary laser in m)
waveS = 830 * 10**-9 ## wavelength of secondary laser in m)
repR = 10000 ## Hz rep rate or modulation frequency if cw
period = 1/repR
Isat = h*c/(tF*sigS0*waveP)

##### Emission Analyzer ######
# Analyzer = 'yes'
Analyzer = 'no'

###### Laser/excitation parameters for simulation - Adjust as needed #####
########### parameters for pulsed primary laser  ######################    
global pPulseW, IPpulsed

pPulseW = 5. * 10**-11 ### primary laser pulse width
IPpulsed = 88.1* 10**2 ## laser peak intensity (W/cm2)

########### remaining excitation parameters for secondary laser (pulsed or cw) - Adjust as needed ######################
global ISMax, ISPulsed,sPulseW,delay,kexSec
ISMax = 82.607* 10**3 ### cw so avg intensity in W/cm2
ISPulsed = 3.354* 10**3 ###peak intensity
sPulseW = 8.*10**-11
delay = 2.*10**-9

kexSec = kexc(ISMax,sigT1,waveS,Isat)
# ########### The Wiener process parameter. ################################ 
# rc_time = 2.524E-7                #rotational correlation time
# delta = (1577.64*(np.pi/180)**2/rc_time)**0.5
# #delta = 25000.
# #rc_time = 1577.64/(delta**2)
# # rotational correlation time (s) = 1577.64/(delta**2) = (diameter/1.96E-6)**3

# # Number of simulated rotational events
# Event_num = 100000
# # Initial angle
# Ini_angle_x = np.random.rand(1,Event_num)*2*np.pi # random initial angle for each molecule. # x = theta 
# Ini_angle_y = np.random.rand(1,Event_num)*2*np.pi # random initial angle for each molecule. # y = phi

# ########### Time bin ############
time_bin = 'linear'
# time_bin = 'log'

# ########### Dye ################
# dye = 'Rb'
# dye = 'AgNC'
dye = 'mVenus'

# In[] Rotation simulation

start_t1 = time.time() # Stopwatch start

rc_time = 250E-8            #rotational correlation time
# delta = (1577.64*(np.pi/180)**2/rc_time)**0.5
delta = (np.sqrt(2)/3/rc_time)**0.5

### rotational correlation time (s) = 1577.64/(delta**2) = (diameter/1.96E-6)**3

# Number of simulated rotational events
Event_num = int(1E6)
# Initial angle
Ini_angle_x = np.random.rand(1,Event_num)*2*np.pi # random initial angle for each molecule. # x = theta 
Ini_angle_y = np.random.rand(1,Event_num)*2*np.pi # random initial angle for each molecule. # y = phi
# Ini_angle_x = np.zeros([1,Event_num])


# When log-time is selected
if time_bin == 'log':
    # Log time scale setting
    ####### time steps for pulsed simulations #####
    pDeltaT = pPulseW*2#1.*10**-9## time step for pulsed excitation
    ### note: probably don't need to go all the way out to the full period. this is a lot of time steps
    maxTp = period
    #tp = np.arange(0,maxTp,pDeltaT)
    ### define logarthmic bins and multiplier to generate "integral" with same bin width as prompt fluorescenc pulse####
    a1 = 10.**(np.arange(np.log10(pDeltaT*10), np.log10(period)))
    pTimeStep = round(pDeltaT*1E10,1)
    a2 = np.arange(1,10,pTimeStep)
    tpTemp = np.outer(a1, a2).flatten()
    ### limit this to go up to the period, but not beyond so we get the right number of bins####
    tpLog = [0]
    for t in tpTemp:
        if t < maxTp:
            tpLog.append(t)
    del tpTemp
     
    # Seperate tpLog into n parts (n = a1). Each part has a fixed step size (i.e. dt)
    tp_all = []
    for i in range(len(a1)):
        temp = len(a2)
        if i == 0:
            tp = tpLog[i*temp:(i+1)*temp+1]
        else:
            tp = tpLog[i*temp+1:(i+1)*temp+1]
        tp_all.append(tp)
            
    # Time info
    for i in range(len(tp_all)):
    # Total time.
        T = tp_all[i][-1]-tp_all[i][0]
    # Number of steps in each linear time bin period
        N = len(tp_all[i])-1
    # Time step size
        dt = T/N
    
        if i == 0:
            tempall_x = r_brownian(Ini_angle_x, N, dt, delta, Event_num)
            tempall_y = r_brownian(Ini_angle_y, N, dt, delta, Event_num)    # rotational brownian motion for tp1 
        else:
            a = tempall_x[-1,:].reshape((1,Event_num))                      # For tp2 and other regions, 
            newtemp_x = r_brownian(a, N+1, dt, delta, Event_num)            # let the last angle in previous region to be 
            tempall_x = np.vstack([tempall_x,newtemp_x[1:,:]])              # the initial angle for the current region.
            b = tempall_y[-1,:].reshape((1,Event_num))                      # To  prevent angle repeat at the junction,  
            newtemp_y = r_brownian(b, N+1, dt, delta, Event_num)            # let molecules rotate one more step (using N+1 for steps),
            tempall_y = np.vstack([tempall_y,newtemp_y[1:,:]])              # drop the initial angle, and then stack the angle array after previous region.
       
    x =np.asarray(tempall_x)
    y =np.asarray(tempall_y)


# When linear-time is selected
elif time_bin == 'linear':
    tpline = np.arange(0, 6E-6, 50E-9)    
    tpLog = tpline  # This helps me to switch between log time scale and linear time scale easier
    
    
    T = tpline
    N = len(T)-1
    dt = T[-1]/N
    
    x = r_brownian(Ini_angle_x, N, dt, delta, Event_num)
    y = r_brownian(Ini_angle_y, N, dt, delta, Event_num)

# Convert arbitrary angle to be in range between -180 to 180
def degree(ang_array):
    result = np.asarray(ang_array)
    result = result/np.pi*360
    if (result.ndim) == 1:
        for i in range(len(result)):    # number of row in result
            if result[i]>=0:
                result[i] = result[i]%360
                if result[i]> 180:
                    result[i] = -360+result[i]
            else:
                result[i]= result[i]%-360
                if result[i]< -180:
                    result[i] = result[i]+360
                elif result[i] == -180:
                    result[i] = 180
    elif (result.ndim) > 1:
        for i in range(len(result[:,0])):    # number of row in result
            for j in range(len(result[0,:])):  # number of column in result
                if result[i,j]>=0:
                    result[i,j] = result[i,j]%360
                    if result[i,j]> 180:
                        result[i,j] = -360+result[i,j]
                else:
                    result[i,j]= result[i,j]%-360
                    if result[i,j]< -180:
                        result[i,j] = result[i,j]+360
                    elif result[i,j] == -180:
                        result[i,j] = 180
    return result

plt.plot(tpLog,degree(x[:,0]), label = 'theta')
plt.plot(tpLog,degree(y[:,0]), label = 'phi')
plt.xlabel('time (s)', fontsize=16)
plt.ylabel('angle', fontsize=16)
plt.ylim(-180,180)
plt.legend(loc= 'upper right')
plt.title('Rotational correlation time: ' + str(round(rc_time*1E9,1))+ ' ns')



end_t1 = time.time() # Stopwatch stop
print('processing time(rotation) =', 
      int((end_t1-start_t1)//3600), ':', int((end_t1-start_t1)%3600//60), ':', int((end_t1-start_t1)%60)) # print stopwatch result


del start_t1, end_t1


# In[] Start a new OADF simulation: reset emission intensities
Ipara = np.zeros([len(tpLog)])
Iperp = np.zeros([len(tpLog)])

# IPri  = np.zeros([len(tpLog)])
# IPripara = np.zeros([len(tpLog)])
# IPriperp = np.zeros([len(tpLog)])

counter = 0

# In[] OADFA simulation
start_t3 = time.time() # Stopwatch start

target_num = 10E6

if Analyzer == 'yes':
    while counter*Event_num < target_num:
        Ipara, Iperp, counter = OADFA_Analyzer_Simu(tpLog, Event_num, x, Ipara, Iperp, counter)
    Itot2 = Ipara + 2*Iperp 
elif Analyzer == 'no':
    while counter*Event_num < target_num:
        Ipara, Iperp, counter = OADFA_Simu(tpLog, Event_num, x, Ipara, Iperp, counter)
    Itot2 = Ipara + Iperp 

end_t3 = time.time() # Stopwatch stop
print('processing time (final) =', 
      int((end_t3-start_t3)//3600), ':', int((end_t3-start_t3)%3600//60), ':', int((end_t3-start_t3)%60)) # print stopwatch result

del start_t3, end_t3
    
    
    
OadFA = np.zeros(len(tpLog))
# PriFA = np.zeros(len(t_series))
for i in range(len(Ipara)):
    if Ipara[i] and Iperp[i]>0:
        OadFA[i] = (Ipara[i]-Iperp[i])/Itot2[i]
# for i in range(len(IPripara)):
#     if IPripara[i] and IPriperp[i]>0:
#         PriFA[i] = (IPripara[i]-IPriperp[i])/(IPripara[i]+2*IPriperp[i])

OadFA[np.isnan(OadFA)] = 0     # convert nan value in OadFA to zero

plt.plot(tpLog*1E6, Itot2, label ='I$_{total}$')
plt.plot(tpLog*1E6, Ipara, label ='I$_\parallel$')
plt.plot(tpLog*1E6, Iperp, label ='I$_\perp$')

# plt.xscale("log",base = 10)
plt.yscale("log",base = 10)
# plt.xlim(0,30)
plt.xlabel('time (\u03BCs)', fontsize = 12)
plt.ylabel('Photon Counts', fontsize = 12)
plt.title('Simulated time-resolved FA (\u03B8 ='+ str(round(rc_time*10**9, 2)) +' ns)')

### add inset ###
# plt.axes([.45, .39, .42, .46], facecolor='floralwhite') #add inset plot
# plt.plot(tpLog*1E6, Itot2, label ='I$_{total}$')
# plt.plot(tpLog*1E6, Ipara, label ='I$_\parallel$')
# plt.plot(tpLog*1E6, Iperp, label ='I$_\perp$')
# plt.yscale("log",base = 10)
# plt.xlim(-0.01,0.65)
# plt.ylim(1E2,1E4)
### inset part ends ###

plt.legend()
plt.show()

# plt.plot(tpLog, ksec_para, label ='ksec_para')
# plt.plot(tpLog, ksec_perp, label ='ksec_perp')
# plt.legend()
# plt.show()

# ksec_r = (ksec_para-ksec_perp)/(ksec_para+1*ksec_perp)
# ksec_r[np.isnan(ksec_r)] = 0     # convert nan value in OadFA to zero

# plt.plot(tpLog, ksec_r, label ='ksec_r')
# plt.legend()
# plt.show()


fitfrom = 3
fitto = 120
assum_I0 = 2 # Assumed decay starting time point
if fitfrom < assum_I0:
    assum_I0 = fitfrom

plt.plot(tpLog*1E6, OadFA, label='Simulated OADFA (\u03B8 = '+ str(round(rc_time*1E6, 3))+ '\u03BCs)')

### Single exponential fitting ###
popt6, pcov6 = curve_fit(expo, tpLog[fitfrom-assum_I0:fitto-assum_I0], OadFA[fitfrom:fitto], 
                          bounds = ([max(OadFA[fitfrom:fitfrom+3])*0.1, 1E-12, -5E-1], 
                                    [1.6, 1.5E-3,  1E-1]))

plt.plot(tpLog[assum_I0:fitto]*1E6,expo(tpLog[0:(fitto-assum_I0)], *popt6),
          label='Fitting OADFA (r0 = '+ str(round(popt6[0], 3))+
          '; \u03B8 = '+ str(round(popt6[1]*1E6, 3))+'\u03BCs)', color = 'r')

### Bi-exponential fitting ###
# popt7, pcov7 = curve_fit(biexpo, tpLog[0:fitto-fitfrom], OadFA[fitfrom:fitto] 
#                           ,bounds = ([max(OadFA[fitfrom:fitfrom+3])*0.8, 1E-12, max(OadFA[fitfrom:fitfrom+3])*0.3, 1E-12, -1E-2], 
#                                     [0.3, 1E13, 0.2, 1E-3,  1E-2]))
# plt.plot(tpLog[fitfrom:fitto],biexpo(tpLog[0:(fitto-fitfrom)], *popt7),
#           label='Fitting OadFA (r0 = '+ str(round(popt7[0], 3))+
#           '; \u03B8 = '+ str(round(popt7[1]*1E6, 3))+'\u03BCs)', color = 'g')


plt.plot(tpLog*1E6, [0 for i in range(len(tpLog))], color = 'gray', linestyle='dashed')
plt.xlabel('time (\u03BCs)', fontsize = 12)
plt.ylabel('Anisotropy', fontsize = 12)
plt.title('Simulated time-resolved OADFA ('+ str(counter) +' M molecules)')
plt.legend(loc='upper right')
plt.ylim(-0.4, 0.6)

plt.show()

p_sigma = np.sqrt(np.diag(pcov6))
diameter = ((popt6[1]*1E9)**(1/3))*1.98
dia_error = abs(1.98/3*(popt6[1]*1E9)**(1/3-1)*(p_sigma[1]*1E9)) # Look up error propagation formula

# print(p_sigma)  # Standard deviation of fitting value
print('r0 =', round(popt6[0], 3),'\u00B1',round(p_sigma[0],3))  
print('\u03B8 =', round(popt6[1]*1E6, 3),'\u00B1',round(p_sigma[1]*1E6,3), '\u03BCs')
print('diameter = ', round(diameter, 2) ,'\u00B1',round(dia_error,2) ,'nm')  
print('***No rebinning***')

# In[] plot with residuals

residuals = OadFA[fitfrom:fitto]-expo(tpLog[fitfrom-assum_I0:fitto-assum_I0], *popt6)

fig, axs = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]})
axs[0].scatter(tpLog*1E6, OadFA, label='Simulated OADFA')
axs[0].plot(tpLog[assum_I0:fitto]*1E6,expo(tpLog[0:(fitto-assum_I0)], *popt6),
          label='Fitting OADFA', color = 'r', linestyle='dashed')

axs[0].plot(tpLog*1.2E6-0.1, [0 for i in range(len(tpLog))], color = 'gray', linestyle='dashed')
axs[0].set_xlabel('time (\u03BCs)', fontsize = 12)
axs[0].set_ylabel('Anisotropy', fontsize = 12)
axs[0].set_xlim(-0.05, 2.55)
axs[0].set_ylim(-0.22,0.45)
# axs[0].set_title('Simulated time-resolved OADFA ('+ str(counter) +' M molecules)')
axs[0].legend(loc='upper right')
# residual plot
axs[1].plot(tpLog*1.2E6-0.1, [0 for i in range(len(tpLog))], color = 'gray', linestyle='dashed')
axs[1].scatter(tpLog[fitfrom:fitto]*1E6, residuals, label = 'Residuals')
axs[1].set_xlim(-0.05, 2.55)
axs[1].set_ylim(-0.2, 0.2)
axs[1].set_xlabel('Time (\u03BCs)',fontsize=12)
axs[1].set_ylabel('Residuals')
plt.show()



 # In[] smoothing and differential of Int
import pandas as pd

def smooth(x, window_size):
    x_series = pd.Series(x)
    windows = x_series.rolling(window_size)
    moving_averages = windows.mean()
    moving_averages = moving_averages.tolist()
    without_nans = moving_averages[window_size - 1:]
    return without_nans

Oadf_start = 4
avg_win = 10
start1 = int(Oadf_start + round(avg_win/2,0))
stop1 = int(round(avg_win/2,0))

Ipara_sm = smooth(Ipara[Oadf_start:], avg_win)
Iperp_sm = smooth(Iperp[Oadf_start:], avg_win)

plt.plot(tpLog*1E9, Ipara, label ='Ipara')
plt.plot(tpLog[start1:-(stop1-1)]*1E9, Ipara_sm, label ='Ipara_sm')

plt.plot(tpLog*1E9, Iperp, label ='Iperp')
plt.plot(tpLog[start1:-(stop1-1)]*1E9, Iperp_sm, label ='Iperp_sm')

plt.legend()
plt.yscale("log",base = 10)
plt.show()
''' Another example
### Note: this still need to be modified. 
###       It assumes everything before data[0] and after data[-1] to be zero
###       Besides, np.cumsum is 30-40 times faster than np.convolve

def smooth2(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

'''
diff_tpLog = np.diff(tpLog)
diff_Ipara = np.diff(Ipara_sm)
diff_Iperp = np.diff(Iperp_sm)

cal_kpara = (-diff_Ipara/diff_tpLog[start1:-stop1+1])/Ipara_sm[:-1]
cal_kperp = (-diff_Iperp/diff_tpLog[start1:-stop1+1])/Iperp_sm[:-1]


plt.plot(tpLog[start1:-stop1-1]*1E9, cal_kpara[:-1], label = 'cal_kpara')
plt.plot(tpLog[start1:-stop1-1]*1E9, cal_kperp[:-1], label = 'cal_kperp')

# plt.xlim(0,10000)
plt.xlabel('Time(ns)')
plt.ylabel('dI/dt')
plt.legend()
plt.show()

popt2, pcov2 = curve_fit(expo, tpLog[0:-(start1+stop1+1)]*1E9, cal_kpara[:-1],
                         bounds = ([max(cal_kpara[:5]*0.8), 1E3, -1E7], [max(cal_kpara[:5]*1.2), 1E12, 1E1])) 
plt.plot(tpLog[start1:-stop1-1]*1E9, expo(tpLog[0:-(start1+stop1+1)]*1E9, *popt2),label = 'expo fit')
plt.plot(tpLog[start1:-stop1-1]*1E9, cal_kpara[:-1], label = 'diff_Iperp')
plt.legend()
plt.show()


rk = (cal_kpara-cal_kperp)#/(cal_kpara+2*cal_kperp)
plt.plot(tpLog[start1:-stop1-1]*1E9, rk[:-1], label = 'rk')
popt3, pcov3 = curve_fit(expo, tpLog[0:-(start1+stop1)]*1E9, rk) 
plt.plot(tpLog[start1:-stop1]*1E9, expo(tpLog[0:-(start1+stop1)]*1E9, *popt3),label = 'expo fit')

plt.legend()
# plt.xlim(0,10000)

print(popt3)

# In[] Save result
import os
import tkinter as tk
from tkinter.filedialog import askdirectory

root = tk.Tk()
root.withdraw()
file_path = askdirectory()
os.chdir(file_path)



Ani = pd.DataFrame(data = [])
columns=['tpLog', 'Ipara', 'Iperp', 'OADFA']
tempdata = [tpLog, Ipara, Iperp, OadFA]
for i in range(len(tempdata)):
    Ani[columns[i]] = tempdata[i]

parameters = pd.DataFrame({'name': ['Dye', 'QYF', 'tF', 'QYT', 'QYT2', 'tT', 'tT2', 
                                    'epS0', 'sigT1', 'sigT2', 'QYrev1', 'QYrev2', 
                                    'waveP', 'waveS', 'repR', 'pPulseW', 'IPpulsed', 
                                  'ISMax', 'ISPulsed', 'sPulseW', 'delay', 'rc_time', 
                                  'Event_num', 'counter'],
                          'data': [dye, QYF, tF, QYT, QYT2, tT, tT2, epS0, sigT1, sigT2, 
                                  QYrev1, QYrev2, waveP, waveS, repR, pPulseW, IPpulsed, 
                                  ISMax, ISPulsed, sPulseW, delay, rc_time, Event_num, counter]})

writer = pd.concat([Ani, parameters], axis=1)
writer.to_csv(dye+' FA simulation '+ str(counter) +'M molecules (rc_time='+ 
              str(round(rc_time*1E9))+'ns).csv', index=False)


# In[] read saved file
import tkinter as tk
from tkinter.filedialog import askdirectory
from tkinter import filedialog

root = tk.Tk()
root.withdraw()
file = filedialog.askopenfilename()

data = pd.read_csv(file, header = 0)
tpLog = data.iloc[:, 0].to_numpy(dtype='float')
Ipara = data.iloc[:, 1].to_numpy(dtype='float')
Iperp = data.iloc[:, 2].to_numpy(dtype='float')
OadFA = data.iloc[:, 3].to_numpy(dtype='float')
dye =  data.iloc[0,5]

[QYF, tF, QYT, QYT2, tT, tT2, epS0, sigT1, sigT2, 
 QYrev1, QYrev2, waveP, waveS, repR, pPulseW, IPpulsed, 
 ISMax, ISPulsed, sPulseW, delay, rc_time] = [float(i) for i in data.iloc[1:22,5]]

Event_num = int(data.iloc[22,5])
counter = int(data.iloc[23,5])

global h,c,nA
h = 6.626*10**-34
c = 2.998*10**8
nA = 6.022 * 10**23
sigS0 = epS0/nA *1000*np.log(10)
epT1 = sigT1/np.log(10)/1000*nA ### dark state extinction coefficient
epT2 = sigT2/np.log(10)/1000*nA ### dark state extinction coefficient
period = 1/repR
Isat = h*c/(tF*sigS0*waveP)