#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 19 18:39:10 2021

@author: FMagnani

"""

import numpy as np
import scipy.io
import tensorflow as tf

from Allen_Cahn_PINN import Allen_Cahn_PINN

#%%

if __name__ == "__main__": 
        
    q = 100
    layers = [1, 200, 200, 200, 200, q+1]
    lb = np.array([-1.0])
    ub = np.array([1.0])
    
    N = 200
    
    data = scipy.io.loadmat('../Data/AC.mat')
    
    t = data['tt'].flatten()[:,None] # T x 1
    x = data['x'].flatten()[:,None] # N x 1
    Exact = np.real(data['uu']).T # T x N
    
    idx_t0 = 20
    idx_t1 = 180
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
    




















    