#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 16:22:29 2021

@author: FMagnani

"""

import numpy as np
import tensorflow as tf
import scipy.io
from pyDOE import lhs 

from Allen_Cahn_Continuous_PINN import Allen_Cahn_Continuous_PINN
from ACC_plotting import plot_results

import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '../Utils/')
from plotting import plot_Adam_history


#%%

if __name__ == "__main__":  
    
    # Set random seed
    np.random.seed(1234)
    tf.random.set_seed(1234)
    
    # Domain bounds
    lb = np.array([-1.0, 0.0])
    ub = np.array([1.0, 1.0])

    N0 = 50     # Number of training pts from x=0
    N_b = 50    # Number of training pts from the boundaries
    N_f = 20000 # Number of training pts from the inside - collocation pts

    ########################################
    ##   DATA PREPARATION                 ##
    ########################################
     
    # Import data
    data = scipy.io.loadmat('../Data/AC.mat')

    
    t = data['tt'].flatten()[:,None] # T x 1
    x = data['x'].flatten()[:,None] # N x 1
    Exact = np.real(data['uu']).T # T x N
    
    # x is too big to be handled (512) so it is reduced by half (to 256)
    # Only even entries are taken
    x = x[ np.arange(0,x.shape[0],2) ,:]
    
    # Creation of the 2D domain
    X, T = np.meshgrid(x,t)
    
    # The whole domain flattened, on which the final prediction will be made
    X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
    u_star = Exact.T.flatten()[:,None]
    
    # Choose N0 training points from x and the corresponding u, v at t=0
    idx_x = np.random.choice(x.shape[0], N0, replace=False)
    x0 = x[idx_x,:]
    u0 = Exact.T[idx_x,0:1]
    
    # Choose N_b training points from the time
    idx_t = np.random.choice(t.shape[0], N_b, replace=False)
    tb = t[idx_t,:]
    
    # Latin Hypercube Sampling of N_f points from the interior domain
    X_f = lb + (ub-lb)*lhs(2, N_f)

    # 2D locations on the domain of the boundary training points
    X0 = np.concatenate((x0, 0*x0), 1) # (x0, 0)
    X_lb = np.concatenate((0*tb + lb[0], tb), 1) # (lb[0], tb)
    X_ub = np.concatenate((0*tb + ub[0], tb), 1) # (ub[0], tb)
                   
    # Recap
    
    # Initial condition pts
    # shape = (N0,1)
    x0 = X0[:,0:1]
    t0 = X0[:,1:2]

    # Boundary pts used for constraint
    # shape = (N_b,1)
    x_lb = X_lb[:,0:1]
    t_lb = X_lb[:,1:2]

    x_ub = X_ub[:,0:1]
    t_ub = X_ub[:,1:2]
        
    # Anchor pts for supervised learning
    # shape = (N_f,1)
    x_f = X_f[:,0:1]
    t_f = X_f[:,1:2]    
    
    # All these are numpy.ndarray with dtype float64
    
    # Conversion to tensors. Recall to WATCH inside a tape
    x0 = tf.convert_to_tensor(x0[:,0])
    t0 = tf.convert_to_tensor(t0[:,0])
    u0 = tf.convert_to_tensor(u0[:,0])
    x_lb = tf.convert_to_tensor(x_lb[:,0])
    t_lb = tf.convert_to_tensor(t_lb[:,0])
    x_ub = tf.convert_to_tensor(x_ub[:,0])
    t_ub = tf.convert_to_tensor(t_ub[:,0])
    x_f = tf.convert_to_tensor(x_f[:,0])
    t_f = tf.convert_to_tensor(t_f[:,0])
    X_star = tf.convert_to_tensor(X_star)


    layers = [1,100,100,100,100,1]
    model = Allen_Cahn_Continuous_PINN(x0, u0, x_ub, x_lb, t_ub, x_f, t_f, X_star, ub, lb, layers)

#%%

    ### TRAINING ###

    adam_iterations = 2  # Number of training steps 
    lbfgs_max_iterations = 2 # Max iterations for lbfgs
    
    Adam_hist = model.train(adam_iterations, lbfgs_max_iterations)
        
    
#%%    
        
    # final prediction
    u_pred = model.model(X_star)
                
    # final error
    u_pred = tf.reshape(u_pred, shape=(51456,1))

    error_u = np.linalg.norm(u_star-u_pred,2)/np.linalg.norm(u_star,2)
    print('Error u: %e' % (error_u))


#%%

    # Plotting
    
    fig_res = plot_results(u_pred, u_star, x,t,x0,tb,lb,ub, x_f,t_f)

    fig_res.show()

    fig_loss, ax_loss = plt.subplots(1,1)
    plot_Adam_history(ax_loss, Adam_hist)






















