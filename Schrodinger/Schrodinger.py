#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 18:24:19 2021

@author: FMagnani

"""

import numpy as np
import tensorflow as tf
import scipy.io
from pyDOE import lhs 

#%%

if __name__ == "__main__":  
    
    # Set random seed
    np.random.seed(1234)
    tf.random.set_seed(1234)
    
    # use float64 by default
    tf.keras.backend.set_floatx("float64")
    
    # Domain bounds
    lb = np.array([-5.0, 0.0])
    ub = np.array([5.0, np.pi/2])

    N0 = 50     # Number of training pts from x=0
    N_b = 50    # Number of training pts from the boundaries
    N_f = 20000 # Number of training pts from the inside

    ########################################
    ##   DATA PREPARATION                 ##
    ########################################
     
    # Import data
    data = scipy.io.loadmat('../Data/NLS.mat')

    
    t = data['tt'].flatten()[:,None]
    x = data['x'].flatten()[:,None]
    Exact = data['uu']
    Exact_u = np.real(Exact)
    Exact_v = np.imag(Exact)
    Exact_h = np.sqrt(Exact_u**2 + Exact_v**2)
    
    # Disretization of the domain
    X, T = np.meshgrid(x,t)
    
    X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
    u_star = Exact_u.T.flatten()[:,None]
    v_star = Exact_v.T.flatten()[:,None]
    h_star = Exact_h.T.flatten()[:,None]
    
    # Choose N0 training points from x and the corresponding u, v
    idx_x = np.random.choice(x.shape[0], N0, replace=False)
    x0 = x[idx_x,:]
    u0 = Exact_u[idx_x,0:1]
    v0 = Exact_v[idx_x,0:1]
    
    # Choose N_b training points from the time
    idx_t = np.random.choice(t.shape[0], N_b, replace=False)
    tb = t[idx_t,:]
    
    # Latin Hypercube Sampling of N_f points from the interior domain
    X_f = lb + (ub-lb)*lhs(2, N_f)

    # Locations on the domain of the boundary training points
    X0 = np.concatenate((x0, 0*x0), 1) # (x0, 0)
    X_lb = np.concatenate((0*tb + lb[0], tb), 1) # (lb[0], tb)
    X_ub = np.concatenate((0*tb + ub[0], tb), 1) # (ub[0], tb)
                       
    x0 = X0[:,0:1]
    t0 = X0[:,1:2]

    x_lb = X_lb[:,0:1]
    t_lb = X_lb[:,1:2]

    x_ub = X_ub[:,0:1]
    t_ub = X_ub[:,1:2]
        
    x_f = X_f[:,0:1]
    t_f = X_f[:,1:2]    

    
    






















