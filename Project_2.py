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
global n, N, J, kb, h, tau, J_eff, t_final

#Physical constants
J = 1.           #Coupling constant
kb = 1.          #Boltzmann constant
h = 0.           #External magnetic field
Tc = 0.44*kb/J   #Predictad critical temperature
T = Tc           #Temperature: low for T<J/4
dT = 0.03         #Step size in temperature (for temperature variation only)
def tau(T):
    tau = kb*T/J     #Reduced temperature
    return tau
def J_eff(T):
    J_eff = J/(kb*T) #Effective coupling constant
    return(J_eff)
    
#Computational parameters
n = 20           #Number of spin sites in one direction
N = n**2         #Number of spin sites
state = 2         #State of the computation: which output is wanted?
                  # 0 = visualization
                  # 1 = magnetization with T variation
                  # 2 = magnetization as function of time
drawtime = 1000   #Draw after every 'drawtime' spinflips (for state 0)
temptime = 3*N     #Amount of time-steps after which temperature is changed
if state == 1:
    t_final = int(temptime*np.floor(T/dT))  #Amount of time-steps (# of spins flipped)
    print("t_final=", t_final)
elif state == 2:
    t_final = 1000*N   # Number of MCS steps
else:
    t_final = 100000 #Amount of time-steps (# of spins flipped)

#Fill an array uniform random with up and down (-1 and 1) spins
S_init = np.random.choice([-1,1],size=(n,n),p=[0.5,0.5])

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
#Flip one spin from i to j and see if energy gets higher/lower
#If lower, keep it. If higher, keep it with probability P = exp(-beta(Hj-Hi))
def spin_flip(S,T):
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
    plt.ion() # Set plot to animated
    #Make the plot
    ax = plt.axes()
    data, = [plt.matshow(S, fignum=0)]
    for i in range(t_final):
        S = spin_flip(S,T)
        data.set_data(S)
        if i%drawtime == 0:
            plt.draw()

#Variation of nett magnetization with temperature
elif state == 1:
    print("Calculating Magnetisation")
    M = np.zeros((t_final/temptime), dtype = float)
    M_T = np.zeros((t_final/temptime),dtype = float)
    for i in range(t_final):
        S = spin_flip(S,T)
        if (i+1)%temptime == 0:
            M[i/temptime] = M_total(S)
            M_T[i/temptime] = T
            T -= dT
    plt.xlabel('T[K]')
    plt.ylabel('M')
    plt.plot(M_T,M)
    plt.show()

#Plot magnetization as a function of k_b*T/J

#Plot magnetization as a function of time
elif state ==2:
    M = np.zeros((t_final/N), dtype = float)
    for i in range(t_final):
        S = spin_flip(S,T)
        if i%10*N==0:
            M[i/N]=M_total(S)
    plt.plot(M)
    plt.xlabel("MCS steps")
    plt.ylabel("M")
    plt.show()
            
   

#Plot the specific heat as a function of reduced temperature

#Measure stoptime
stoptime = time.clock() - starttime
print(stoptime)
