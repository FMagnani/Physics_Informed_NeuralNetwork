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

#Plotting
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import griddata



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

    model = NN.Schrod_PINN_LBFGS(x0, u0, v0, x_ub, x_lb, t_ub, x_f, t_f, X_star, ub, lb)

    
#%%

    ########################################
    ##   MODEL TRAINING AND PREDICTION    ##
    ########################################

    n_iterations = 1  # Number of training steps 
    
##### Training
    model.train(n_iterations)
            
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

    ######################################################################
    ############################# Plotting ###############################
    ######################################################################    
    
    # U_pred = griddata(X_star, tf.reshape(u_pred, [-1]), (X, T), method='cubic')
    # V_pred = griddata(X_star, tf.reshape(v_pred, [-1]), (X, T), method='cubic')
    H_pred = griddata(X_star, tf.reshape(h_pred, [-1]), (X, T), method='cubic')

    # FU_pred = griddata(X_star, f_u_pred, (X, T), method='cubic')
    # FV_pred = griddata(X_star, f_v_pred, (X, T), method='cubic')     
    
    
    X0 = tf.stack([x0,0*x0],axis=1) # (x0, 0)
    X_lb = tf.stack([0*tb + lb[0], tb], axis=1) # (lb[0], tb)
    X_ub = tf.stack([0*tb + ub[0], tb], axis=1) # (ub[0], tb)
    X_u_train = np.vstack([X0, X_lb[:,:,0], X_ub[:,:,0]])

    

    ###########  h(t,x)  ##################    
    
    fig1, ax1 = plt.subplots(1,1)
    
    gs0 = gridspec.GridSpec(1, 2)
    gs0.update(top=1-0.06, bottom=1-1/3, left=0.15, right=0.85, wspace=0)
    ax1 = plt.subplot(gs0[:, :])
    
    h = ax1.imshow(H_pred.T, interpolation='nearest', cmap='YlGnBu', 
                  extent=[lb[1], ub[1], lb[0], ub[0]], 
                  origin='lower', aspect='auto')
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig1.colorbar(h, cax=cax)
    
    ax1.plot(X_u_train[:,1], X_u_train[:,0], 'kx', label = 'Data (%d points)' % (X_u_train.shape[0]), markersize = 4, clip_on = False)
    
    line = np.linspace(x.min(), x.max(), 2)[:,None]
    ax1.plot(t[75]*np.ones((2,1)), line, 'k--', linewidth = 1)
    ax1.plot(t[100]*np.ones((2,1)), line, 'k--', linewidth = 1)
    ax1.plot(t[125]*np.ones((2,1)), line, 'k--', linewidth = 1)    
    
    ax1.set_xlabel('$t$')
    ax1.set_ylabel('$x$')
#    leg = ax1.legend(frameon=False, loc = 'best')
#    plt.setp(leg.get_texts(), color='w')
    ax1.set_title('$|h(t,x)|$', fontsize = 10)
    

    

    ########   h(t,x) slices ##################    
 
    gs1 = gridspec.GridSpec(1, 3)
    gs1.update(top=1-1/3, bottom=0, left=0.1, right=0.9, wspace=0.5)
    
    ax = plt.subplot(gs1[0, 0])
    ax.plot(x,Exact_h[:,75], 'b-', linewidth = 2, label = 'Exact')       
    ax.plot(x,H_pred[75,:], 'r--', linewidth = 2, label = 'Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$|h(t,x)|$')    
    ax.set_title('$t = %.2f$' % (t[75]), fontsize = 10)
    ax.axis('square')
    ax.set_xlim([-5.1,5.1])
    ax.set_ylim([-0.1,5.1])
    
    ax = plt.subplot(gs1[0, 1])
    ax.plot(x,Exact_h[:,100], 'b-', linewidth = 2, label = 'Exact')       
    ax.plot(x,H_pred[100,:], 'r--', linewidth = 2, label = 'Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$|h(t,x)|$')
    ax.axis('square')
    ax.set_xlim([-5.1,5.1])
    ax.set_ylim([-0.1,5.1])
    ax.set_title('$t = %.2f$' % (t[100]), fontsize = 10)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.8), ncol=5, frameon=False)
    
    ax = plt.subplot(gs1[0, 2])
    ax.plot(x,Exact_h[:,125], 'b-', linewidth = 2, label = 'Exact')       
    ax.plot(x,H_pred[125,:], 'r--', linewidth = 2, label = 'Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$|h(t,x)|$')
    ax.axis('square')
    ax.set_xlim([-5.1,5.1])
    ax.set_ylim([-0.1,5.1])    
    ax.set_title('$t = %.2f$' % (t[125]), fontsize = 10)
    
    

