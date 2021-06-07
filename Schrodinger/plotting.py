#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 15:25:01 2021

@author: FMagnani

"""

import numpy as np
import tensorflow as tf

#Plotting
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import griddata

#%%

######################################################################
############################# Plotting ###############################
######################################################################    
    
 
    
def plot_results(h_pred, h_exact, x,t,x0,tb,lb,ub, error_mode=False):

    # Create the grid of the domine
    X, T = np.meshgrid(x,t)
    X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))

    H_pred = griddata(X_star, h_pred[:,0], (X, T), method='cubic')
    H_true = griddata(X_star, h_exact[:,0], (X, T), method='cubic')
    H_error = np.abs(H_pred - H_true)
    
    if error_mode:
        cmap = 'inferno'
        H_to_plot = H_error
 
    else:
        cmap='YlGnBu'
        H_to_plot = H_pred
    
    # Data location on the boundaries
    X0 = tf.stack([x0,0*x0],axis=1) # (x0, 0)
    X_lb = tf.stack([0*tb + lb[0], tb], axis=1) # (lb[0], tb)
    X_ub = tf.stack([0*tb + ub[0], tb], axis=1) # (ub[0], tb)
    X_u_train = np.vstack([X0, X_lb[:,:,0], X_ub[:,:,0]])    

    ###########  h(t,x)  ##################    
    
    fig1, ax1 = plt.subplots(1,1)
    
    if not error_mode:
        # Select first row of the figure
        gs0 = gridspec.GridSpec(1, 2)
        gs0.update(top=1-0.06, bottom=1-1/3, left=0.15, right=0.85, wspace=0)
        ax1 = plt.subplot(gs0[:, :])
    
    h = ax1.imshow(H_to_plot.T, interpolation='nearest', cmap=cmap, 
                   extent=[lb[1], ub[1], lb[0], ub[0]], 
                   origin='lower', aspect='auto')
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig1.colorbar(h, cax=cax)
    
    if not error_mode:
        # Print lines corresponding to the time slices
        ax1.plot(X_u_train[:,1], X_u_train[:,0], 'kx', label = 'Data (%d points)' % (X_u_train.shape[0]), markersize = 4, clip_on = False)
    
        line = np.linspace(x.min(), x.max(), 2)[:,None]
        ax1.plot(t[75]*np.ones((2,1)), line, 'k--', linewidth = 1)
        ax1.plot(t[100]*np.ones((2,1)), line, 'k--', linewidth = 1)
        ax1.plot(t[125]*np.ones((2,1)), line, 'k--', linewidth = 1)    
    
    # Title and labels
    ax1.set_xlabel('$t$')
    ax1.set_ylabel('$x$')
    ax1.set_title('$|h(t,x)|$', fontsize = 10)
        

    ########   h(t,x) slices ##################    
    
    if not error_mode:
    
        # Select secpnd row of the figure
        gs1 = gridspec.GridSpec(1, 3)
        gs1.update(top=1-1/3, bottom=0, left=0.1, right=0.9, wspace=0.5)
    
        # First slice
        ax = plt.subplot(gs1[0, 0])
        ax.plot(x,H_true[75,:], 'b-', linewidth = 2, label = 'Exact')       
        ax.plot(x,H_pred[75,:], 'r--', linewidth = 2, label = 'Prediction')
        ax.set_xlabel('$x$')
        ax.set_ylabel('$|h(t,x)|$')    
        ax.set_title('$t = %.2f$' % (t[75]), fontsize = 10)
        ax.axis('square')
        ax.set_xlim([-5.1,5.1])
        ax.set_ylim([-0.1,5.1])
    
        # Second slice
        ax = plt.subplot(gs1[0, 1])
        ax.plot(x,H_true[100,:], 'b-', linewidth = 2, label = 'Exact')       
        ax.plot(x,H_pred[100,:], 'r--', linewidth = 2, label = 'Prediction')
        ax.set_xlabel('$x$')
        ax.set_ylabel('$|h(t,x)|$')
        ax.axis('square')
        ax.set_xlim([-5.1,5.1])
        ax.set_ylim([-0.1,5.1])
        ax.set_title('$t = %.2f$' % (t[100]), fontsize = 10)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.8), ncol=5, frameon=False)
        
        # Third slice
        ax = plt.subplot(gs1[0, 2])
        ax.plot(x,H_true[125,:], 'b-', linewidth = 2, label = 'Exact')       
        ax.plot(x,H_pred[125,:], 'r--', linewidth = 2, label = 'Prediction')
        ax.set_xlabel('$x$')
        ax.set_ylabel('$|h(t,x)|$')
        ax.axis('square')
        ax.set_xlim([-5.1,5.1])
        ax.set_ylim([-0.1,5.1])    
        ax.set_title('$t = %.2f$' % (t[125]), fontsize = 10)
        
    return fig1
        
