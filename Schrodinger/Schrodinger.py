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
import time
from tqdm import tqdm

import NeuralNets as NN
from Utilities import net_uv, net_f_uv

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

    # Casting to float32, the output dtype of the predictions
    # x0 = tf.cast(x0, dtype='float32')
    # t0 = tf.cast(x0, dtype='float32')
    # x_lb = tf.cast(x_lb, dtype='float32')
    # t_lb = tf.cast(t_lb, dtype='float32')
    # x_ub = tf.cast(x_ub, dtype='float32')
    # t_ub = tf.cast(t_ub, dtype='float32')
    # x_f = tf.cast(x_f, dtype='float32')
    # t_f = tf.cast(t_f, dtype='float32')
    # u0 = tf.cast(u0, dtype='float32')
    # v0 = tf.cast(v0, dtype='float32')

    
#%%

    def create_loss(x_lb, t_lb, x_ub, t_ub, x_f, t_f, u0, v0):
                
        def loss():

            X = tf.concat([x0,t0],1)
            
            u0_pred, v0_pred = model(X)
                
            u_lb_pred, v_lb_pred, u_x_lb_pred, v_x_lb_pred = net_uv(x_lb, t_lb)
            u_ub_pred, v_ub_pred, u_x_ub_pred, v_x_ub_pred = net_uv(x_ub, t_ub)

            f_u_pred, f_v_pred = net_f_uv(x_f, t_f)
    
            
            y_Schrodinger = tf.reduce_mean(tf.square(f_u_pred)) + \
                           tf.reduce_mean(tf.square(f_v_pred))

            y_boundary = tf.reduce_mean(tf.square(u_lb_pred - u_ub_pred)) + \
                         tf.reduce_mean(tf.square(v_lb_pred - v_ub_pred)) + \
                         tf.reduce_mean(tf.square(u_x_lb_pred - u_x_ub_pred)) + \
                         tf.reduce_mean(tf.square(v_x_lb_pred - v_x_ub_pred))

            y_supervised = tf.reduce_mean(tf.square(u0 - u0_pred)) + \
                           tf.reduce_mean(tf.square(v0 - v0_pred))

            return y_Schrodinger + y_boundary + y_supervised
        
        return loss

#%%

    ########################################
    ##   MODEL TRAINING AND PREDICTION    ##
    ########################################

    model = NN.neural_net(ub, lb)

    n_iterations = 10  # Number of training steps 
    
    # Optimizer and loss
    
    optimizer = tf.keras.optimizers.Adam()
    
    loss = create_loss(x_lb, t_lb, x_ub, t_ub, x_f, t_f, u0, v0)

    # Training
    
    start_time = time.time()    
    #Train step
    for _ in tqdm(range(n_iterations)):
        optimizer.minimize(loss, model.trainable_variables)
    elapsed = time.time() - start_time                
    print('Training time: %.4f' % (elapsed))
            
    # final prediction
    u_pred, v_pred = model(X_star)
    f_u_pred, f_v_pred = net_f_uv(X_star[:,0:1], X_star[:,1:2])    
    h_pred = np.sqrt(u_pred**2 + v_pred**2)
                
    # final error
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
    
    U_pred = griddata(X_star, tf.reshape(u_pred, [-1]), (X, T), method='cubic')
    V_pred = griddata(X_star, tf.reshape(v_pred, [-1]), (X, T), method='cubic')
    H_pred = griddata(X_star, tf.reshape(h_pred, [-1]), (X, T), method='cubic')

    # FU_pred = griddata(X_star, tf.reshape(f_u_pred, [-1]), (X, T), method='cubic')
    # FV_pred = griddata(X_star, tf.reshape(f_v_pred, [-1]), (X, T), method='cubic')     
    
    
    X0 = np.concatenate((x0, 0*x0), 1) # (x0, 0)
    X_lb = np.concatenate((0*tb + lb[0], tb), 1) # (lb[0], tb)
    X_ub = np.concatenate((0*tb + ub[0], tb), 1) # (ub[0], tb)
    X_u_train = np.vstack([X0, X_lb, X_ub])

    

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
    leg = ax1.legend(frameon=False, loc = 'best')
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
    
    

#%%




















