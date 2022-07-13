#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 15:00:36 2022

@author: rolerich
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(4,4))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(-5,5)
ax.set_ylim(-5,5)
ax.set_zlim(0,2)
ax.scatter(2,3,0.5) # plot one point 
ax.scatter(-2,-1,1.7) # plot another point 
# plt.show()
plt.savefig("two_3D_points.png",dpi=300) # saving plot with high resolution
