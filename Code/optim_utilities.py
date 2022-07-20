#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A collection of utilities for testing optimizers

@author: Rodolphe Le Riche, Brian Dédji Whannou
"""
import numpy as np

def record_hist(rec : dict, f : float, x : np.array, time: int, 
                key_f : str, key_x : str, key_time : str) -> dict :
    """ generic function to update history records """
    if key_f not in rec:
        # first recording, create fields
        rec[key_f] = [f]
        rec[key_time] = [time]
        rec[key_x] = x
    else:
        rec[key_f].append(f)
        rec[key_time].append(time)
        rec[key_x] = np.vstack([rec[key_x],x])     
            
    return rec

def record_best(rec : dict, fbest : float, xbest : np.array, time: int, printlevel: int) -> dict:
    """ create and update records of best so far """
    rec["x_best"]=xbest
    rec["f_best"]=fbest
    if printlevel>0:
        rec = record_hist(rec=rec,f=fbest,x=xbest,time=time,
                          key_f="hist_f_best",key_x="hist_x_best",key_time="hist_time_best")
        
    return rec

def record_any(rec : dict, f : float, x : np.array, time: int, printlevel: int) -> dict:
    """ create and update record of any point during the search """
    if "time_used" not in rec:
        rec["time_used"]=1
    else:
        rec["time_used"]+=1
    if printlevel > 1:
        rec = record_hist(rec=rec,f=f,x=x,time=time,
                          key_f="hist_f",key_x="hist_x",key_time="hist_time")
        
    return rec