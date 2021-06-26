#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 11:57:42 2021

@author: FMagnani

"""

import numpy as np
import tensorflow as tf
import scipy.io
import matplotlib.pyplot as plt

from Schrodinger_Discrete_PINN import Schrodinger_PINN
from plotting import plot_slice

import sys
sys.path.insert(0, '../Utils/')
from plotting import plot_Adam_history

#%%

if __name__ == "__main__":  
    
    # Set random seed
    np.random.seed(1234)
    tf.random.set_seed(1234)

    # Network architecture    
    q = 100
    layers = [1, 200, 200, 200, 200, 2*(q+1)]
    
    # Domain bounds
    lb = np.array([-5.0, 0.0])          # left bottom corner
    ub = np.array([5.0, np.pi/2])       # right upper corner

    N = 200     # Number of training pts from x=0 <- The only needed!

    ###    DATA PREPARATION    ###
     
    # Import data
    data = scipy.io.loadmat('../Data/NLS.mat')

    
    t = data['tt'].flatten()[:,None]
    x = data['x'].flatten()[:,None]
    Exact = data['uu']
    Exact_u = np.real(Exact).T
    Exact_v = np.imag(Exact).T

    Exact_h = np.sqrt(Exact_u**2 + Exact_v**2)

    # t slices: 75, 100, 125

    idx_t0 = 0
    idx_t1 = 10
    dt = t[idx_t1] - t[idx_t0]

    idx_x = np.random.choice(Exact_u.shape[1], N, replace=False) 

    # Initial data - u
    noise_u0 = 0.0
    x0 = x[idx_x,:]
    u0 = Exact_u[idx_t0:idx_t0+1,idx_x].T
    u0 = u0 + noise_u0*np.std(u0)*np.random.randn(u0.shape[0], u0.shape[1])

    # Initial data - v
    noise_v0 = 0.0
    x0 = x[idx_x,:]
    v0 = Exact_v[idx_t0:idx_t0+1,idx_x].T
    v0 = v0 + noise_v0*np.std(u0)*np.random.randn(u0.shape[0], u0.shape[1])

    # Test data
    x_star = x
    
    # Conversion to tensors    
    x0 = tf.convert_to_tensor(x0)
    u0 = tf.convert_to_tensor(u0)
    v0 = tf.convert_to_tensor(v0)
    x_star = tf.convert_to_tensor(x_star)

    model = Schrodinger_PINN(x0, u0,v0, layers, dt, lb, ub, q)



#%%

    ###    TRAINING    ###

    adam_iterations = 300     # Number of training steps 
    lbfgs_max_iterations = 300 # Max iterations for lbfgs
    
    Adam_hist = model.train(adam_iterations, lbfgs_max_iterations)


#%%

    U1_pred, V1_pred = model.predict(x_star)
    h_pred = np.sqrt(U1_pred[:, -1:]**2 + V1_pred[:, -1:]**2)

#%%

    ###    PLOTTING    ###
    
    fig, [ax_u, ax_v, ax_h] = plt.subplots(1,3)
    
    plot_slice(ax_u, Exact_u,U1_pred[:,-1:], idx_t1, x_star,t)
    plot_slice(ax_v, Exact_v,V1_pred[:,-1:], idx_t1, x_star,t)
    plot_slice(ax_h, Exact_h,h_pred, idx_t1, x_star,t)
    plt.show()

    fig_loss, ax_loss = plt.subplots(1,1)
    plot_Adam_history(ax_loss, Adam_hist)





