#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 13:07:55 2021

@author: FMagnani

"""

# import numpy as np

import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec
# from mpl_toolkits.axes_grid1 import make_axes_locatable


#%%

def plot_slice(Exact_h,h_pred, idx_t1, x,t):
    
    fig, ax = plt.subplots(1,1)
    
    ax.plot(x,Exact_h[idx_t1,:], 'b-', linewidth = 2, label = 'Exact')       
    ax.plot(x,h_pred, 'r--', linewidth = 2, label = 'Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$|h(t,x)|$')    
    ax.set_title('$t = %.2f$' % (t[idx_t1]), fontsize = 10)
    ax.axis('square')
    ax.set_xlim([-5.1,5.1])
    ax.set_ylim([-0.1,5.1])

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    





