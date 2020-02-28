# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 23:04:38 2017
@author: Diego S
Numerov method for Dirac equation: 3d step potential
"""

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn')
from scipy.optimize import brentq
from scipy import integrate

l, L, Vo = 0.0, 5.0, 1.5 # angular momentum, potential width, potential value
x_max = 3*L # maximum value of x (the choice depends of the value of L)
h = 0.0001 # step value for solve the ode
e = 0.01 # step value for energies of the eigenstates
N = int(x_max/h) # number of iterations
u = np.zeros(N,dtype="double") # change of variable radial part wave function g, so that u=g*r
u[0], u[1] = 0, h # initial conditions


def V(x,E):
    # (minus) effective potential function
    if (x > L):
        return -E**2+1+l*(l+1)/x**2
    else:
        if (x == 0):
            return -(E+Vo)**2+1+l*(l+1)/h**4
        else:
            return -(E+Vo)**2+1+l*(l+1)/x**2

def wavef(E):
    # calculates the u(x_max)
    for i in range(1, N-1):
        u[i+1] = (2 * (1 + 5 * (h**2) * V(i*h, E)/12) * u[i] - (1 - (h**2) * V((i-1)*h, E)/12) * u[i-1])/(1 - (h**2) * V((i+1) * h, E)/12)
    return u[-1]

def find_E_levels(energies, u_max):
    # find all zeroes in u(x_max) = 0
    zeroes = []
    s = np.sign(u_max)
    for i in range(len(u_max)-1):
        if (s[i]+s[i+1] == 0): #sign change
            zero = brentq(wavef, energies[i], energies[i+1])
            zeroes.append(zero)
    return zeroes

def main():
    energies = np.linspace(-1, 1, int(2/e)) # vector of energies where we look for the stable states
    u_max = []  # vector for values of the wave function at x_max
    
    for E in energies:
        u_max.append(wavef(E)) # for each energy find the psi at x_max
    
    E_levels = find_E_levels(energies, u_max)
    
    # plot the wavefunctions for the eigenstates)
    x = np.linspace(0, x_max, N)
    y = np.zeros(N)
    y[0] = 1/(h**2)
    for i in range(1, N):
        y[i] = 1/x[i]
    plt.figure()
    print ("The energies for the bound states are: ")

    i = 1
    for E in E_levels:
        print (i, "E =", "%.4f"%E)
        i = i + 1
        wavef(E)
        norm = np.sqrt(integrate.simps(u**2, x)) # finding normalization factor
        u_norm = u/norm
        g = np.multiply(u_norm, y) # recovering the radial part g=u/r
        g[0] = g[1]
        print("g(x_max):", g[-1])
        print ("Norm ->", integrate.simps(u_norm**2, x))
        plt.plot(x, g, label="%.4f"%E)
    
    plt.legend(loc="upper right")
    plt.xlabel('$r$')
    plt.ylabel('$g(r)$')
    plt.tight_layout()
    plt.savefig('wavefunction.pdf')
    plt.show()

main()
