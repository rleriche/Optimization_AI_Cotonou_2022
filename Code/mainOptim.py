#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main script for testing optimizers

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

# algorithms settings
budgetMax = 10

# optimize
xbest,fbest = random_opt(func=fun, LB=LB, UB=UB, max_budget=budgetMax)