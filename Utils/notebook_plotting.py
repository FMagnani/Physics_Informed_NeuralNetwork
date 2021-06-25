#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 15:18:21 2021

@author: FMagnani

"""

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

#%%

def plot_results_2steps(U1_pred_training,idx_t1,
                        U2_pred,idx_t2,
                        u0,idx_t0,x0,
                        Exact,x,t,lb,ub,x_star):
    
    fig, ax = plt.subplots(1,1)
    ax.axis('off')
    
    ####### Row 0: h(t,x) ##################    
    gs0 = gridspec.GridSpec(1, 1)
    gs0.update(top=1-0.06, bottom=1-1/2 + 0.1, left=0.15, right=0.85, wspace=0)
    ax = plt.subplot(gs0[:, :])
    
    # True solution on the full domine
    h = ax.imshow(Exact.T, interpolation='nearest', cmap='seismic', 
                  extent=[t.min(), t.max(), np.min(x_star), np.max(x_star)], 
                  origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)
    
    # Time lines
    line = np.linspace(x.min(), x.max(), 2)[:,None]
    ax.plot(t[idx_t0]*np.ones((2,1)), line, 'w-', linewidth = 1)
    ax.plot(t[idx_t1]*np.ones((2,1)), line, 'w-', linewidth = 1)
    ax.plot(t[idx_t2]*np.ones((2,1)), line, 'w-', linewidth = 1)
    
    ax.set_xlabel('$t$')
    ax.set_ylabel('$x$')
    ax.set_title('$u(t,x)$', fontsize = 10)

    ####### Row 1: h(t,x) slices ##################    
    gs1 = gridspec.GridSpec(1, 3)
    gs1.update(top=1-1/2-0.05, bottom=0.15, left=0.15, right=0.85, wspace=0.5)
 
    # Initial conditions - data and true 
    ax = plt.subplot(gs1[0, 0])
    ax.plot(x,Exact[idx_t0,:], 'b-', linewidth = 2)  # True 
    ax.plot(x0, u0, 'rx', linewidth = 2, label = 'Data') # Data     
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(t,x)$')    
    ax.set_title('$t = %.2f$' % (t[idx_t0]), fontsize = 10)
    ax.set_xlim([lb-0.1, ub+0.1])

    # First prediction
    ax = plt.subplot(gs1[0, 1])
    ax.plot(x,Exact[idx_t1,:], 'b-', linewidth = 2, label = 'Exact') # True
    ax.plot(x0, U1_pred_training, 'rx', linewidth = 2, label = 'Data') # Data    
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(t,x)$')    
    ax.set_title('$t = %.2f$' % (t[idx_t0]), fontsize = 10)
    ax.set_xlim([lb-0.1, ub+0.1])
    ax.legend(loc='upper center', bbox_to_anchor=(0.8, -0.3), ncol=2, frameon=False)

    # Second prediction
    ax = plt.subplot(gs1[0, 2])
    ax.plot(x,Exact[idx_t2,:], 'b-', linewidth = 2, label = 'Exact') # True
    ax.plot(x_star, U2_pred, 'r--', linewidth = 2, label = 'Prediction') # Pred
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(t,x)$')    
    ax.set_title('$t = %.2f$' % (t[idx_t1]), fontsize = 10)    
    ax.set_xlim([lb-0.1, ub+0.1])
    
    return fig
    