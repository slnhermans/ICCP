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
import sys
sys.setrecursionlimit(10000)

#Definition of parameters
global n, N, J, kb, tau, J_eff, t_final

#Physical constants
J = 1.           #Coupling constant
kb = 1.          #Boltzmann constant
h = 0.           #External magnetic field
dh = 0.01        #Step size in h (for h variation only)
Tc = 0.44*kb/J   #Predictad critical temperature
T = 3.           #Start Temperature: low for T<J/
Tf = 10.*Tc       #Final temperature
dT = 0.04         #Step size in temperature (for temperature variation only)
sign = 1.         #Can be 1 or -1; determines sign of all spins in the initial matrix.
def tau(T):
    tau = kb*T/J     #Reduced temperature
    return tau
def J_eff(T):
    J_eff = J/(kb*T) #Effective coupling constant
    return(J_eff)
    
#Computational parameters
n = 32           #Number of spin sites in one direction
N = n**2         #Number of spin sites
state = 4         #State of the computation: which output is wanted?
                  # 0 = visualization
                  # 1 = magnetization with T variation
                  # 2 = magnetization as function of time
                  # 3 = energy with T variation
                  # 4 = Specific heat with T variation
                  # 5 = Magnetic Susceptibility 
TorH = 0          #For variation: are we varying T or h?
                  # 0 = varying temperature
                  # 1 = varying external magnetic field
wolff = 1         # 1 for using the Wolff algorithm; 0 for not using it.
drawtime = 1   #Draw after every 'drawtime' spinflips (for state 0)
temptime = 1000     #Amount of time-steps after which temperature is changed (relaxtion time)
if state == 1 or state == 3 or state == 5 or state == 4:
    t_final = int(temptime*np.floor(Tf/dT))  #Amount of time-steps (# of spins flipped)
    print("t_final=", t_final)
elif state == 2:
    t_final = 1000*N   # Number of MCS steps
else:
    t_final = 2000 #Amount of time-steps (# of spins flipped)

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
    dM = S[x,y]/N
    return S, dE, dM

#################################################################################
#################################################################################
#Flip one spin from -1 to 1 and see if energy gets higher/lower
#If lower, keep it. If higher, keep it with probability P = exp(-beta(Hj-Hi))
#If lower, keep it. If higher, keep it with probability P = exp(-beta(Hj-Hi))
def growcluster(x, y, S, Cluster,P):
    S[x,y] = -S[x,y] #Flip spin at location
    ClusterSpin = S[x,y] #The spin of the cluster
    Cluster[x,y] = 1 #Add spin to cluster
    for [a, b] in [ [(x+1)%n,y], [(x-1)%n,y], [x,(y+1)%n], [x,(y-1)%n] ]:
      if Cluster[a,b] != 1 and S[a,b] != ClusterSpin:
            tryadd(a, b, S, Cluster, ClusterSpin,P)
    return S, Cluster

def tryadd(a, b, S, Cluster, ClusterSpin,P):
    if np.random.choice([0,1],p=[1-P, P]) == 1:
        growcluster(a, b, S, Cluster, P)
    return S, Cluster

def spin_flip_wolff(S,T,h):
    Cluster = np.zeros((n,n),dtype = int) #Matrix that says for every analog in S if it is in the cluster
    x, y = np.random.randint(0,n,size=2)
    #With a chance P, perimeter spins are added to the cluster
    P = 1 - np.exp(-2.*J/(kb*T))
    S = growcluster(x, y, S, Cluster,P)[0]
    return S
#################################################################################
##################################OUTPUT FUNCTIONS########################################
#################################################################################
#################################################################################

#Visualization of te spin matrix
def visualization(S,T,h,wolff):
    S = S_init_rand
    plt.ion() # Set plot to animated
    #Make the plot
    ax = plt.axes()
    data, = [plt.matshow(S, fignum=0)]
    for i in range(t_final):
        if wolff == 1:
            S = spin_flip_wolff(S,T,h)
        else:    
            S = spin_flip(S,T,h)[0]
        data.set_data(S)
        if i%drawtime == 0:
            plt.draw()

#Variation of nett magnetization with temperature or magnetic field
def magnetization(S,T,h,wolff,TorH,dT,dh):
    print("Calculating Magnetisation [T]")
    M = np.zeros((t_final/temptime), dtype = float)
    M_x = np.zeros((t_final/temptime),dtype = float)
    for i in range(t_final):
        if wolff == 1:
            S = spin_flip_wolff(S,T,h)
        else:    
            S = spin_flip(S,T,h)[0]
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
def magnetization_time(S,T,h,wolff):
    print("Calculating Magnetisation [time]")
    M = np.zeros((t_final/N), dtype = float)
    for i in range(t_final):
        if wolff == 1:
            S = spin_flip_wolff(S,T,h)
        else:    
            S = spin_flip(S,T,h)[0]
        if i%10*N==0:
            M[i/N]=M_total(S)
    plt.plot(M)
    plt.xlabel("MCS steps")
    plt.ylabel("E")
    plt.show()

#Variation of total energy with temperature
def total_E(S,T,h,wolff,TorH,dT,dh):
    print("Calculating Total energy [T]")
    E = np.zeros((t_final/temptime), dtype = float)
    E_x = np.zeros((t_final/temptime),dtype = float)
    for i in range(t_final):
        if wolff == 1:
            S = spin_flip_wolff(S,T,h)
        else:    
            S = spin_flip(S,T,h)[0]
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
def specific_heat(S,T,h):
    print("Calculating Specific heat [T]")
    E_T = np.zeros((t_final/temptime),dtype = float)
    C = np.zeros((t_final/temptime),dtype = float)
    E_init=E_total(S)
    E_avg = 0
    E2_avg = 0
    for i in range(t_final):
        S, dE, dM = spin_flip(S,T,h)
        E_avg += (dE+E_init)
        E2_avg += (dE+E_init)**(2)
        if (i+1)%temptime == 0:
            fluc_E2 = E2_avg/temptime - (E_avg/temptime)**2
            E_T[i/temptime] = tau(T)
            C[i/temptime]= fluc_E2/(kb*(E_T[i/temptime])**2)
            T += dT
            E_init = E_avg/temptime
            E2_avg = 0
            E_avg = 0
            print(i/temptime)
    plt.title("Specific heat as a function of reduced Temperature")
    plt.xlabel('kb T/J')
    plt.ylabel('C')
    plt.plot(E_T,C)
    plt.show()

# Calculate the Magnetic Susceptibility as a function of temperature
def magnetic_susceptibility(S,T,h,dT):
    print("Calculating Magnetic Susceptibility")
    M_T = np.zeros((t_final/temptime),dtype = float)
    Xi = np.zeros((t_final/temptime),dtype = float)
    M_init=M_total(S)
    M_avg = 0
    M2_avg = 0
    for i in range(t_final):
        S, dE, dM = spin_flip(S,T,h)
        M_avg += (dM+M_init)
        M2_avg += (dM+M_init)**(2)
        if (i+1)%temptime == 0:
            fluc_M2 = M2_avg/temptime - (M_avg/temptime)**2
            M_T[i/temptime] = T
            Xi[i/temptime]= fluc_M2/(kb*T)
            T += dT
            M_init = M_avg/temptime
            M2_avg = 0
            M_avg = 0
            print(i/temptime)
    plt.title("Magnetic Susceptibility as temperature")
    plt.xlabel('T')
    plt.ylabel('Xi')
    plt.plot(M_T,Xi)
    plt.show()

# Calculate the the correlation length as a function of temperature

###############################################################################
##################################MAIN RUN#####################################
###############################################################################

print("start")
S = S_init #Initiate the data
if state == 0:
    visualization(S,T,h,wolff)
elif state == 1:
    magnetization(S,T,h,wolff,TorH,dT,dh)
elif state ==2:    
    magnetization_time(S,T,h,wolff)
elif state == 3:
    total_E(S,T,h,wolff,TorH,dT,dh)
elif state == 4:
    specific_heat(S,T,h)
elif state == 5:
    magnetic_susceptibility(S,T,h,dT)


#Measure stoptime
stoptime = time.clock() - starttime
print(stoptime)
