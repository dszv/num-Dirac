# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 23:04:38 2017

@author: Diego S

Numerov method for Dirac equation: 3d step potential
"""

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from scipy.optimize import brentq
from scipy import integrate

l, L, Vo = 0.0, 10.0, 1.5 #angular momentum, potential width, potential value
x_max = 2.25*L #maximum value of x (the choice depends of the value of L)
h = 0.001 #step value for solve the ode
e = 0.01 #step value for energies of the eigenstates
N = int(x_max/h) #number of iterations
psi = np.zeros(N,dtype="double") #wave function
psi[0], psi[1] = 0, h #initial conditions


#Effective potential function
def V(x,E):
    #Effective potential function
    if x > L:
        return -E**2+1+l*(l+1)/x**2
    else:
        return -(E+Vo)**2+1+l*(l+1)/x**2
    
def Wavef(E):
    #Calculates the psi(3L) = 0
    for i in range(2, N):
        psi[i] = (2 * (1 + 5 * (h**2) * V(i*h, E)/12) * psi[i-1] - (1 - (h**2) * V((i-1)*h, E)/12) * psi[i-2])/(1 - (h**2) * V((i+1) * h, E)/12)
    return psi[-1]

def find_E_levels(energies,psi_max):
    #Find all zeroes in psi(3L) = 0
    zeroes = []
    s = np.sign(psi_max)
    for i in range(len(psi_max)-1):
        if s[i]+s[i+1] == 0: #sign change
            zero = brentq(Wavef, energies[i], energies[i+1])
            zeroes.append(zero)
    return zeroes


def main():
    energies = np.linspace(-1, 1, int(2/e)) #vector of energies where we look for the stable states
    psi_max = []  # vector for values of the wave function at x = 3L
    
    for E in energies:
        psi_max.append(Wavef(E)) #for each energy find the psi_max at xmax
    
    E_levels = find_E_levels(energies, psi_max) #now find the energies where psi_max = 0
    
    # Plot the wavefunctions for the eigenstates)
    x = np.linspace(0, x_max, N)
    plt.figure()
    print ("Energies for the bound states are: ")
    i = 1
    
    for E in E_levels:
        print (i, "E =", "%.3f"%E)
        i = i + 1
        Wavef(E)
        Norm = np.sqrt(integrate.simps(psi**2, x)) #finding normalization factor
        psi_norm = psi/Norm
        print("Psi_norm(max):", psi_norm[-1])
        print ("Norm ->", integrate.simps(psi_norm**2, x))
        plt.plot(x, psi_norm, label="%.3f"%E)
    
    plt.legend(loc="upper right")
    plt.xlabel('$r$')
    plt.ylabel('$u(r)$')
    plt.tight_layout()
    plt.savefig('wavefunction.pdf')
    plt.show()

main()
