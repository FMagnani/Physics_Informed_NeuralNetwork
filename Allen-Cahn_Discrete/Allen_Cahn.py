#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 19 18:39:10 2021

@author: FMagnani

"""

import numpy as np
import scipy.io
import tensorflow as tf
import matplotlib.pyplot as plt

from Allen_Cahn_PINN import Allen_Cahn_PINN
from ACD_plotting import plot_results

import sys
sys.path.insert(0, '../Utils/')
from plotting import plot_loss_history

#%%

if __name__ == "__main__": 
        
    ###    MAIN    ###
    
    # Set random seed
    np.random.seed(1234)
    tf.random.set_seed(1234)
    
    q = 10
    layers = [1, 200, 200, 200, 200, q+1]
    lb = np.array([-1.0])
    ub = np.array([1.0])
    
    N = 200
    
    data = scipy.io.loadmat('../Data/AC.mat')
    
    t = data['tt'].flatten()[:,None] # T x 1
    x = data['x'].flatten()[:,None] # N x 1
    Exact = np.real(data['uu']).T # T x N
    
    idx_t0 = 20
    idx_t1 = 10
    dt = t[idx_t1] - t[idx_t0]
    
    # Initial data
    noise_u0 = 0.0
    idx_x = np.random.choice(Exact.shape[1], N, replace=False) 
    x0 = x[idx_x,:]
    u0 = Exact[idx_t0:idx_t0+1,idx_x].T
    u0 = u0 + noise_u0*np.std(u0)*np.random.randn(u0.shape[0], u0.shape[1])
    
    # Test data
    x_star = x

    # Conversion to tensors    
    x0 = tf.convert_to_tensor(x0)
    u0 = tf.convert_to_tensor(u0)
    x_star = tf.convert_to_tensor(x_star)

    model = Allen_Cahn_PINN(x0, u0, layers, dt, lb, ub, q)
    
#%%

    ###    TRAINING    ###

    # GPU: lbfgs - 100 its/16 sec

    adam_iterations = 15      # Number of training steps 
    lbfgs_max_iterations = 15 # Max iterations for lbfgs
    
    Adam_loss_hist, LBFGS_loss_hist = model.train(adam_iterations, lbfgs_max_iterations)
        


#%%

    ###    PREDICTION    ###

    U1_pred = model.predict(x_star)

    error = np.linalg.norm(U1_pred[:,-1] - Exact[idx_t1,:], 2)/np.linalg.norm(Exact[idx_t1,:], 2)
    print('Error: %e' % (error))

#%%

    ###    PLOTTING    ###

    fig = plot_results(U1_pred, Exact,t,x_star,x,idx_t0,idx_t1,x0,u0,lb,ub)
    
    fig1, ax1 = plt.subplots(1,1)
    plot_loss_history(ax1, Adam_loss_hist, LBFGS_loss_hist)
    
    
    


    