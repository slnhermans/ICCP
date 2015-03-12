#Delft University of Technology
#International Course in Computational Physics
#Assignment 2: Ising model
#Authors: Emma C. Gerritse and Sophie L. N. Hermans
import numpy as np

#Definition of parameters
n = 10       #Number of spin sites in one direction
N = n**2     #Number of spin sites
J = 1.       #Coupling constant
kb = 1.      #Boltzmann constant
h = 0.       #External magnetic field
T = 293.     #Temperature: low for T<J/4

#Fill an array uniform random with up and down (-1 and 1) spins
S_init = np.random.choice([-1,1],size=(n,n),p=[0.5,0.5])

#Calculate the total energy (as efficiently as possible!!!)
def E_total():
    
    return E_total

#Calculate the total magnetization
def M_total(S,N):
    M_total = np.sum(S)/(N*1.)
    return M_total

print(M_total(S_init,N))
#Implement periodic boundary conditions

#Plot magnetization as a function of k_b*T/J
