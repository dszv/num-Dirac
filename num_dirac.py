# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 23:04:38 2017

@author: Imhotep

Numerov method for Dirac equation: 3d step potential
"""

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from scipy.optimize import brentq

l = float(input("Angular momentum l:"))
L = float(input("Width of the potential:"))
Vo = float(input("Value of the potential:"))
N = int(input("Number of steps (~10000):"))
h = float(3*L/N)
psi = np.zeros(N,dtype="float64") #wave function
psi[0] = 0
psi[1] = h

def V(x,E):
    """
    Effective potential function.
    """
    if x > L:
        return -E**2+1+l*(l+1)/x**2
    else:
        return -(E+Vo)**2+1+l*(l+1)/x**2
    
def Wavefunction(energy):
    """
    Calculates wave function psi for the given value
    of energy E and returns value at point xmax
    """
    global psi
    global E
    E=energy
    for i in range(2,N):
        psi[i]=(2*(1+5*(h**2)*V(i*h,E)/12)*psi[i-1]-(1-(h**2)*V((i-1)*h,E)/12)*psi[i-2])/(1-(h**2)*V((i+1)*h,E)/12)       
    return psi[-1]

def find_energy_levels(x,y):
    """
    Gives all zeroes in y = psi_max, x=en
    """
    zeroes = []
    s = np.sign(y)
    for i in range(len(y)-1):
        if s[i]+s[i+1] == 0: #sign change
            zero = brentq(Wavefunction, x[i], x[i+1])
            zeroes.append(zero)
    return zeroes

def main():
    
    energies = np.linspace(-1,1,int(10*Vo))   # vector of energies where we look for the stable states
    psi_max = []  # vector of wave function at x = 3L for all of the energies in energies
    for energy in energies:
        psi_max.append(Wavefunction(energy))     # for each energy find the the psi_max at xmax
    E_levels = find_energy_levels(energies,psi_max)   # now find the energies where psi_max = 0  
    print ("Energies for the bound states are: ")
    for E in E_levels:
        print ("%.2f" %E)
    # Plot the wavefunctions for first 4 eigenstates
    x = np.linspace(0, 3*L, N)
    plt.figure()
    for E in E_levels:
        Wavefunction(E)
        plt.plot(x, psi, label="E = %.2f"%E)
    plt.legend(loc="upper right")
    plt.xlabel('$r$')
    plt.ylabel('$u(r)$', fontsize = 10)
    plt.savefig('num_dirac.pdf', bbox_inches='tight')
    
if __name__ == "__main__":
    main()


