#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A set of test functions to play with (plot, optimize, ...).

    All test functions are defined in arbitrary dimension.

@author: Rodolphe Le Riche, Brian Dédji Whannou
"""

import numpy as np
from numpy import exp
from numpy import sqrt
from numpy import cos
from numpy import e
from numpy import pi

###########################################
def ackley(x, a=20, b=0.2, c=2*pi):
    """ Ackley's function. Global optimum at (0,...,0). """
    d=len(x)    
    x=6.4*x
    return -a * exp(-b * sqrt((1/d) * np.sum(x**2)))-exp((1/d) * np.sum(cos(c*x))) + e + a

        
###########################################
def sphere(x):
    """" Sphere function. Global optimum at xstar = (1,2,...,dim) """
    d=len(x)
    xstar=np.array(range(1,(d+1)))
    xx=x-xstar
    return np.sum(xx**2)
  
    
###########################################
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
        cond_no = 4 #condition number of Hessian
        eigvals = np.diag(np.linspace(1, cond_no,dim))
        # np.random.seed(1) # to make runs repeatable, 
        # potentially dangerous so I comment for now
        H = np.random.randn(dim, dim)
        Q, R = qr(H)
        # Q = np.diag(np.ones(dim)) # to have axes parallel to the coordinates
        __Hmat__ = Q @ eigvals @ Q.T


    xeff = np.reshape((x - xstar),(dim,1))
    return float(0.5*(xeff.T @ __Hmat__ @ xeff))

###########################################


def rosen(x):
    """
    Rosenbrock function. Global optimum at (1,...,1)
    copied and slightly changed from Sonja Surjanovic and Derek Bingham, 
    Simon Fraser University. 
    For function details and reference information, see:
    http://www.sfu.ca/~ssurjano/
    """
    dim = len(x)
    xi = x[0:(dim-1)]
    xnext=x[1:dim]
    return np.sum(100*(xnext-xi**2)**2 + (xi-1)**2)

###########################################

def L1norm(x):
    """
    non differentiable and convex L1 norm
    useful for regularization

    Parameters
    ----------
    x : np.array
        vector of inputs

    Returns
    -------
    float
        sum_i(abs(x_i))

    """
    return np.sum(np.abs(x))


###########################################

def sphereL1(x: np.array) -> float:
    """
    Regularized sphere function

    Parameters
    ----------
    x : np.array
        vector of input variables.

    Returns
    -------
    float
        sphere(x)+lbda*L1norm(x).

    """    
    lbda = 0.01
    return sphere(x) + lbda*L1norm(x)



