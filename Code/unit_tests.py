#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 16:18:27 2022

scripts for unit tests

@author: Rodolphe Le Riche
"""

import numpy as np
import test_functions as tf
from gradient_descent import get_gradient

#%% check get_gradient
# playing with the following code, I got convinced that epsilon=1.e-7 is a reasonable setting
fun = tf.sphere

def sphere_grad(x):
    """ analytical gradient of the sphere """
    dim = len(x)
    df = np.zeros(shape=(dim,),dtype=float)
    for i in range(dim):
        df[i]=2*(x[i]-(i+1))
        
    return(df)

dim=4
LB = -5*np.ones(dim)
UB = 5*np.ones(dim)
x = np.random.uniform(low=LB,high=UB)
fd_grad = get_gradient(func = fun, x=x,epsilon=1.e-7)
exact_grad = sphere_grad(x)
print("exact gradient =",exact_grad)
print("exact - finite_diff grads =",exact_grad-fd_grad)

#%% 
