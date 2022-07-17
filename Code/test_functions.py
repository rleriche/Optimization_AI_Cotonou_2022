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
    d=len(x)    
    x=6.4*x
    return -a * exp(-b * sqrt((1/d) * np.sum(x**2)))-exp((1/d) * np.sum(cos(c*x))) + e + a

                    
  