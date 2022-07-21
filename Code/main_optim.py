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
import matplotlib.pyplot as plt
import test_functions
from random_search import random_opt
from gradient_descent import gradient_descent

#########################
# function definition
fun = test_functions.quadratic
LB = [-5] * 10
UB = [5] * 10
dim = len(LB)
np.random.seed(123) # useful for repeated runs (quadratic fct or initial random point)

#########################
# algorithms settings
start_x = (1+np.arange(dim))*5/dim
# start_x = np.array([4,4,4,4,4])
# start_x = np.random.uniform(low=LB,high=UB)

budget = 1000*(dim+1)
printlevel = 1  # =0,1,2

#########################
# optimize
# res = random_opt(func=fun, LB=LB, UB=UB, budget=budget, printlevel=printlevel)
res = gradient_descent(func=fun,start_x=start_x, LB=LB,UB=UB,budget=budget,
                       step_factor=0.1,direction_type="momentum",
                       do_linesearch=True,min_step_size=1e-11,
                       min_grad_size=1e-6,inertia=0.9,printlevel=printlevel)

#########################
# reporting
print(f'search stopped after {res["time_used"]} evaluations of f because of {res["stop_condition"]}')
print("best objective function =",res["f_best"])
print("best x =", res["x_best"])
if printlevel > 0:
    fig1, ax1 = plt.subplots()
    plt.yscale("log")
    ax1.plot((res["hist_time_best"]+ [res["time_used"]]) , (res["hist_f_best"] + [res["f_best"]]))
    ax1.set_xlabel("no. calls to f")
    ax1.set_ylabel("f")
    if printlevel > 1:
        ax1.plot(res["hist_time"],res["hist_f"])
    if dim == 2: 
        # 2D contour plot 
        # start drawing the function (necessarily dim==2)
        no_grid = 100
        x1 = np.linspace(start=LB[0], stop=UB[0],num=no_grid)
        x2 = np.linspace(start=LB[1], stop=UB[1],num=no_grid)
        x, y = np.meshgrid(x1, x2)
        xy = np.array([x,y])
        z = np.apply_along_axis(fun,0,xy)
        fig2, ax2 = plt.subplots()
        # fquant = np.quantile(a=res["hist_f_best"],q=[0,0.05,0.1,0.2,0.5,0.7,1])
        fmin = min(res["hist_f_best"])
        fmax = max(res["hist_f_best"])
        fquant = fmin + (np.linspace(start=0,stop=1,num=10)**2)*(fmax-fmin)        
        CS = ax2.contour(x,y,z,levels=fquant)
        ax2.clabel(CS, inline=True, fontsize=10)
        # add history of best points onto it
        if printlevel > 1:
            ax2.plot(res["hist_x"][:,0],res["hist_x"][:,1],"ob",markersize=3)
        ax2.plot(res["hist_x_best"][:,0],res["hist_x_best"][:,1],"or",markersize=4)
        
