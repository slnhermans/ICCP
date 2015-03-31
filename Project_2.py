#Delft University of Technology
#International Course in Computational Physics
#Assignment 2: Ising model
#Authors: Emma C. Gerritse and Sophie L. N. Hermans
###############################################################################
#Program for simulating the nearest neighbour two-dimmensional Ising model on a
#square lattice using the Metropolis Monte Carlo technique.
###############################################################################

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import time

#Definition of parameters
global n, N, J, kb, tau, J_eff, t_final

#Physical constants
J = 1.           #Coupling constant
kb = 1.          #Boltzmann constant
h = 0.           #External magnetic field
dh = 0.01        #Step size in h (for h variation only)
Tc = 0.44*kb/J   #Predictad critical temperature
T = 10**(-10)    #Start emperature: low for T<J/
Tf = 10.*Tc       #Final temperature
dT = 0.01         #Step size in temperature (for temperature variation only)
sign = 1.         #Can be 1 or -1; determines sign of all spins in the initial matrix.
def tau(T):
    tau = kb*T/J     #Reduced temperature
    return tau
def J_eff(T):
    J_eff = J/(kb*T) #Effective coupling constant
    return(J_eff)
    
#Computational parameters
n = 20           #Number of spin sites in one direction
N = n**2         #Number of spin sites
state = 1         #State of the computation: which output is wanted?
                  # 0 = visualization
                  # 1 = magnetization with T variation
                  # 2 = magnetization as function of time
                  # 3 = energy with T variation
TorH = 0          #For variation: are we varying T or h?
                  # 0 = varying temperature
                  # 1 = varying external magnetic field
drawtime = 500   #Draw after every 'drawtime' spinflips (for state 0)
temptime = 5*N     #Amount of time-steps after which temperature is changed
if state == 1 or state == 3:
    t_final = int(temptime*np.floor(Tf/dT))  #Amount of time-steps (# of spins flipped)
    print("t_final=", t_final)
elif state == 2:
    t_final = 1000*N   # Number of MCS steps
else:
    t_final = 20000 #Amount of time-steps (# of spins flipped)

#Fill an array uniform random with up and down (-1 and 1) spins
S_init_rand = np.random.choice([-1,1],size=(n,n),p=[0.5,0.5])
S_init = sign*np.ones((n,n),dtype = float)

#Measure the start time
starttime = time.clock()

###############################################################################
###########################Function definitions################################
###############################################################################

#Calculate the total energy of the system
def E_total(S):
    E_total = 0
    cnt = 0
    for i in range(N):
        E_total -= h * S[i%n,cnt/n] #Due to magnetic field
        E_total -= J * S[i%n,cnt/n] * (S[(i%n+1)%n,cnt/n] + S[i%n,(cnt/n+1)%n]) #Due to spin-spin interaction
        cnt += 1
    return E_total

#Calculate the total magnetization of the system
def M_total(S):
    M_total = np.sum(S)/(N*1.)
    return M_total

############
#Flip one spin from -1 to 1 and see if energy gets higher/lower
#If lower, keep it. If higher, keep it with probability P = exp(-beta(Hj-Hi))
def spin_flip(S,T,h):
    x, y = np.random.randint(0,n,size=2)
    E_old = -h * S[x,y] - J * S[x,y] * (S[(x+1)%n,y] + S[(x-1)%n,y] + S[x,(y+1)%n] + S[x,(y-1)%n] )
    E_new = -h * -S[x,y] - J * -S[x,y] * (S[(x+1)%n,y] + S[(x-1)%n,y] + S[x,(y+1)%n] + S[x,(y-1)%n] )
    dE = E_new - E_old
    if dE <= 0:
        S[x,y] = -S[x,y]
    else:
        P = np.exp(-dE/(kb*T))
        S[x,y] = S[x,y] * np.random.choice([-1,1],p=[P, 1-P])
    return S

#################################################################################
#################################################################################
#Flip one spin from -1 to 1 and see if energy gets higher/lower
#If lower, keep it. If higher, keep it with probability P = exp(-beta(Hj-Hi))
def spin_flip_wolff(S,T,h):
    InCluster = np.zeros((n,n),dtype = int) #Matrix that says for every analog in S if it is in the cluster
    x, y = np.random.randint(0,n,size=2)
    return InCluster
def Wolff_growth(x, y, InCluster):
    spin_flip(S,T,h)
    return S
#################################################################################
#################################################################################


#Calculate the specific heat

#Calculate all critical exponents
#######Make a function for fitting

#Fitting function for y = x^alpha
def polyfit(x,alpha):
    return x**alpha
def crit_exp(x, y, alpha):
    alpha, alpha_err = curve_fit(polyfit, x, y, alpha)
    return alpha, alpha_err
###############################################################################
##################################MAIN RUN#####################################
###############################################################################

S = S_init #Initiate the data
print("start")
#Visualization of te spin matrix
if state == 0:
    S = S_init_rand
    plt.ion() # Set plot to animated
    #Make the plot
    ax = plt.axes()
    data, = [plt.matshow(S, fignum=0)]
    for i in range(t_final):
        S = spin_flip(S,T,h)
        data.set_data(S)
        if i%drawtime == 0:
            plt.draw()

#Variation of nett magnetization with temperature or magnetic field
elif state == 1:
    print("Calculating Magnetisation [T]")
    M = np.zeros((t_final/temptime), dtype = float)
    M_x = np.zeros((t_final/temptime),dtype = float)
    for i in range(t_final):
        S = spin_flip(S,T,h)
        if (i+1)%temptime == 0:
            M[i/temptime] = M_total(S)
            if TorH == 0:
                M_x[i/temptime] = tau(T)
                T += dT
            elif TorH == 1:
                M_x[i/temptime] = h
                h += dh
            print(i/temptime)
    plt.xlabel('kb T/J')
    plt.ylabel('M')
    plt.plot(M_x,M)
    plt.show()

#Plot magnetization as a function of time
elif state ==2:
    print("Calculating Magnetisation [time]")
    M = np.zeros((t_final/N), dtype = float)
    for i in range(t_final):
        S = spin_flip(S,T,h)
        if i%10*N==0:
            M[i/N]=M_total(S)
    plt.plot(M)
    plt.xlabel("MCS steps")
    plt.ylabel("E")
    plt.show()
            
#Variation of total energy with temperature
elif state == 3:
    print("Calculating Total energy [T]")
    E = np.zeros((t_final/temptime), dtype = float)
    E_x = np.zeros((t_final/temptime),dtype = float)
    for i in range(t_final):
        S = spin_flip(S,T,h)
        if TorH == 0:
            E_x[i/temptime] = tau(T)
            T += dT
        elif TorH == 1:
            E_x[i/temptime] = h
            h += dh
        print(i/temptime)
    plt.xlabel('kb T/J')
    plt.ylabel('E')
    plt.plot(E_x,E)
    plt.show()   

#Plot the specific heat as a function of reduced temperature

#Measure stoptime
stoptime = time.clock() - starttime
print(stoptime)
