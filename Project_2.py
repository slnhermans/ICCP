#Delft University of Technology
#International Course in Computational Physics
#Assignment 2: Ising model
#Authors: Emma C. Gerritse and Sophie L. N. Hermans
###############################################################################
#Program for simulating the nearest neighbour two-dimmensional Ising model on a
#square lattice using the Metropolis Monte Carlo technique.
###############################################################################

import numpy as np

#Definition of parameters
global n, N, J, kb, h

n = 10       #Number of spin sites in one direction
N = n**2     #Number of spin sites
J = 1.       #Coupling constant
kb = 1.      #Boltzmann constant
h = 0.       #External magnetic field
T = 293.     #Temperature: low for T<J/4

#Fill an array uniform random with up and down (-1 and 1) spins
S_init = np.random.choice([-1,1],size=(n,n),p=[0.5,0.5])

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
    x, y = np.random.randint(0,n-1,size=2)
    E_old = -h * S[x,y] - J * S[x,y] * (S[(x+1)%n,y] + S[(x-1)%n,y] + S[x,(y+1)%n] + S[x,(y-1)%n] )
    E_new = -h * -S[x,y] - J * -S[x,y] * (S[(x+1)%n,y] + S[(x-1)%n,y] + S[x,(y+1)%n] + S[x,(y-1)%n] )
    dE = E_new - E_old

    if dE <= 0:
        S[x,y] = -S[x,y]
    else:
        P = exp(-dE/(kb*T))
        S[x,y] = S[x,y] * np.random.choice([-1,1],p=[P, 1-P])
    return S
    
###############################################################################
##################################MAIN RUN#####################################
###############################################################################
#Plot magnetization as a function of k_b*T/J
