#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A simple random search within a bounded hypercube

@author: Rodolphe Le Riche, Brian Dédji Whannou
"""
import numpy as np

def random_opt (func: object, LB, UB, max_budget: int = 1e3):
    if len(LB) != len(UB):
        raise Exception("the bounds should have the same nb of components and it is %d and %d" % (len(LB),len(UB)))
    dim=len(UB)
    iteration = 0
    func_value_best_so_far = float("inf")
    position_best_so_far = np.zeros(dim)
    condition = False
    while not condition:
        xnew = np.random.uniform(low=LB,high=UB)
        fnew = func(xnew)
        iteration += 1
        if fnew < func_value_best_so_far:
            func_value_best_so_far = fnew
            position_best_so_far = xnew
            
        condition = iteration > max_budget
        
    return position_best_so_far, func_value_best_so_far
