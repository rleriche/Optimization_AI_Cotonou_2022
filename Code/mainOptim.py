#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
        "hist_f_best", list(float) : history of best so far objective functions
        "hist_time_best", list(int) : times of recordings of new best so far
        "hist_x_best", 2D array : history of best so far points as a matrix, each x is a row


        

@author: Rodolphe Le Riche, Brian Dédji Whannou
"""

import numpy as np
import matplotlib.pyplot as plt
import test_functions
from random_search import random_opt

# function definition
dim = 2
LB = [-5,-5]
UB = [5,5]
fun = test_functions.rosen

#########################
# algorithms settings
budgetMax = 100
printlevel = 1

#########################
# optimize
res = random_opt(func=fun, LB=LB, UB=UB, max_budget=budgetMax, printlevel=printlevel)

#########################
# reporting
print("best objective function =",res["f_best"])
print("best x =", res["x_best"])
if printlevel > 0:
    fig1, ax1 = plt.subplots()
    plt.yscale("log")
    ax1.plot(res["hist_time_best"],res["hist_f_best"])
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
        fquant = np.quantile(a=res["hist_f_best"],q=[0,0.05,0.1,0.2,0.5,0.7,1])
        CS = ax2.contour(x,y,z,levels=fquant)
        ax2.clabel(CS, inline=True, fontsize=10)
        # add history of best points onto it
        ax2.plot(res["hist_x_best"][:,0],res["hist_x_best"][:,1],"ob",markersize=3)