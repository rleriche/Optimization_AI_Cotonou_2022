#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#%% main . TODO : documentation of module and update README
"""
Main script for testing optimizers

Parameters
----------
    printlevel : int, controls how much is recorded during optimization. 
        = 0 for minimum recording (best point found and its obj function value)
        > 0 records history of best points
        > 1 records the entire history of points (memory consuming)
        
    The optimization results are dictionaries with the following key-value pairs:
        "f_best", float : best ojective function found during the search
        "x_best", 1D array : best point found 
        "stop_condition" : str describing why the search stopped
        "time_used" , int : time actually used by search (may be smaller than max budget)
        if printlevel > 0 :
            "hist_f_best", list(float) : history of best so far objective functions
            "hist_time_best", list(int) : times of recordings of new best so far
            "hist_x_best", 2D array : history of best so far points as a matrix, each x is a row
        if printlevel > 1 :
        "hist_f", list(float) : all f's calculated
        "hist_x", 2D array : all x's calculated
        "hist_time", list(int) : times of recording of full history

        

@author: Rodolphe Le Riche, Brian Dédji Whannou
"""

import numpy as np
import test_functions
from random_search import random_opt
from gradient_descent import gradient_descent
from optim_utilities import print_rec

#########################
# function definition
fun = test_functions.rosen
dim = 2
LB = [-5] * dim
UB = [5] * dim
# np.random.seed(123) # useful for repeated runs (quadratic fct or initial random point)

#########################
# algorithms settings
# start_x = np.array([3,2,1,-4.5,4.6,-2,-1,4.9,0,2])
# start_x = (1+np.arange(dim))*5/dim
# start_x = np.array([2.3,4.5])
start_x = np.random.uniform(low=LB,high=UB)

budget = 1000*(dim+1)
printlevel = 1  # =0,1,2 , careful with 2 which is memory consuming

#########################
# optimize
# res = random_opt(func=fun, LB=LB, UB=UB, budget=budget, printlevel=printlevel)
res = gradient_descent(func=fun,start_x=start_x, LB=LB,UB=UB,budget=budget,
                       step_factor=0.1,direction_type="momentum",
                       do_linesearch=True,min_step_size=1e-11,
                       min_grad_size=1e-6,inertia=0.9,printlevel=printlevel)

#########################
# reporting
print_rec(res=res, fun=fun, dim=dim, LB=LB, UB=UB , printlevel=printlevel, logscale = True)

