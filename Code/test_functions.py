#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A set of test functions to play (plot, optimize, ...) with

@author: Rodolphe Le Riche
"""

import numpy as np
from numpy import exp
from numpy import sqrt
from numpy import cos
from numpy import e
from numpy import pi


def ackley(x, a=20, b=0.2, c=2*pi):
    """ Ackley's function. Global optimum at (0,...,0). """
    d=len(x)    
    x=6.4*x
    return -a * exp(-b * sqrt((1/d) * np.sum(x**2)))-exp((1/d) * np.sum(cos(c*x))) + e + a

        
def sphere(x):
    """" Sphere function. Global optimum at xstar. """
    d=len(x)
    xstar=np.array(range(1,(d+1)))
    xx=x-xstar
    return np.sum(xx**2)
  
    
from scipy.linalg import qr

def quadratic(x):
    """ quadratic function. 
         Hessian sampled at first use and saved in global __Hmat__ 
         cond_no sets the condition number of the Hessian.
         Solution at xstar.
    """
    dim = len(x)
    xstar = np.zeros(dim)
    global __Hmat__ 
    update_H = False
    try: 
        if __Hmat__.shape != (dim,dim):
            update_H = True
    except NameError:
        update_H = True
        
    if update_H:
        cond_no = 3 #condition number of Hessian
        eigvals = np.diag(np.linspace(1, cond_no,dim))
        np.random.seed(1)
        H = np.random.randn(dim, dim)
        Q, R = qr(H)
        __Hmat__ = Q @ eigvals @ Q.T
    #     print("new hmat")
    #     print(__Hmat__)     
    # else:
    #     print("old H")
    #     print(__Hmat__)

    xeff = np.reshape((x - xstar),(dim,1))
    return float(0.5*(xeff.T @ __Hmat__ @ xeff))


