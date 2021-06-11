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

import NeuralNets as NN
from plotting import plot_results


#%%

if __name__ == "__main__":  
    
    # Set random seed
    np.random.seed(1234)
    tf.random.set_seed(1234)
    
    # Domain bounds
    lb = np.array([-5.0, 0.0])
    ub = np.array([5.0, np.pi/2])

    N0 = 50     # Number of training pts from x=0
    N_b = 50    # Number of training pts from the boundaries
    N_f = 20000 # Number of training pts from the inside - anchor pts

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
    
    # Creation of the 2D domain
    X, T = np.meshgrid(x,t)
    
    # The whole domain flattened, on which the final prediction will be made
    X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
    u_star = Exact_u.T.flatten()[:,None]
    v_star = Exact_v.T.flatten()[:,None]
    h_star = Exact_h.T.flatten()[:,None]
    
    # Choose N0 training points from x and the corresponding u, v at t=0
    idx_x = np.random.choice(x.shape[0], N0, replace=False)
    x0 = x[idx_x,:]
    u0 = Exact_u[idx_x,0:1]
    v0 = Exact_v[idx_x,0:1]
    
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
    
    # Initial condition pts - the real supervised learning pts
    # Their 'labels' are u0, v0
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
    v0 = tf.convert_to_tensor(v0[:,0])
    x_lb = tf.convert_to_tensor(x_lb[:,0])
    t_lb = tf.convert_to_tensor(t_lb[:,0])
    x_ub = tf.convert_to_tensor(x_ub[:,0])
    t_ub = tf.convert_to_tensor(t_ub[:,0])
    x_f = tf.convert_to_tensor(x_f[:,0])
    t_f = tf.convert_to_tensor(t_f[:,0])
    X_star = tf.convert_to_tensor(X_star)


    layers = [2,100,100,100,100,2]
    model = NN.Schrod_PINN_LBFGS(x0, u0, v0, x_ub, x_lb, t_ub, x_f, t_f, X_star, ub, lb, layers)

    
#%%

    ########################################
    ##   MODEL TRAINING AND PREDICTION    ##
    ########################################

    # 500 ADAM + 1000 LBFGS is enough for a quite satisfactory result
    # Set to 0 the iteration to completely skip that optimization method   

    check_freq = 2  # Number of training steps 
    
    adam_total_it = 6
    adam_groups = int(adam_total_it/check_freq)
    
    lbfgs_max_iterations = 14 # Max iterations for lbfgs
    lbfgs_groups = int(lbfgs_max_iterations/check_freq)
    
##### Training   
    for group in range(adam_groups):
        model.train(check_freq, 0)
        it = (group+1)*check_freq
        w_name = 'checkpoints/adam_'+str(it)
        model.model.save_weights(w_name)

    for group in range(lbfgs_groups):
        model.train(0, check_freq)
        it = (group+1)*check_freq
        w_name = 'checkpoints/lbfgs_'+str(it)
        model.model.save_weights(w_name)
        

#%%    
        
##### final prediction
    u_pred, v_pred, f_u_pred, f_v_pred = model.predict(X_star[:,0], X_star[:,1])
    h_pred = np.sqrt(u_pred**2 + v_pred**2)
                
##### final error
    u_pred = tf.reshape(u_pred, shape=(51456,1))
    v_pred = tf.reshape(v_pred, shape=(51456,1))
    h_pred = tf.reshape(h_pred, shape=(51456,1))

    error_u = np.linalg.norm(u_star-u_pred,2)/np.linalg.norm(u_star,2)
    error_v = np.linalg.norm(v_star-v_pred,2)/np.linalg.norm(v_star,2)
    error_h = np.linalg.norm(h_star-h_pred,2)/np.linalg.norm(h_star,2)
    print('Error u: %e' % (error_u))
    print('Error v: %e' % (error_v))
    print('Error h: %e' % (error_h))


#%%

    # Plotting
    
    fig = plot_results(h_pred, h_star, x,t,x0,tb,lb,ub, x_f,t_f)

#    fig.savefig('results')












