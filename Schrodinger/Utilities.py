#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 20:05:23 2021

@author: FMagnani

"""

import tensorflow as tf


#%%    

# Train step:
    # - Compute the loss (watching)
    # - Compute the gradient of the loss wrt the model weights (trainable vars)
    # - Apply the gradients (optimizer does)
    
# Access to model guaranteed
# Data needed: input and their labels for prediction, iterations, optimizer
def train_step(x0, t0, u0, v0, n_iterations, optimizer):
    
    with tf.GradientTape() as tape:
        
        loss_value = loss(u0,v0, model(x0,t0))
    
    grads = tape.gradients(loss_value, model.trainable.variables)
    
    optimizer.apply_gradients(zip(grads, model.trainable_variables))




# def net_uv(x, t):
    
#     X = tf.concat([x,t],1)
#     X = tf.Variable(X)
    
#     with tf.GradientTape(persistent=True) as tape:
#         u, v = model(X)
         
#     u_X = tape.gradient(u, X)
#     v_X = tape.gradient(v, X)

#     u_x = u_X[:,0:1]
#     v_x = v_X[:,1:2]

#     return u, v, u_x, v_x

# That's not a real function, it has no inputs. It simply carry the computation
# of the derivatives of the neural net model and returns the f_u, f_v values
# NEEDS ACCESS TO MODEL AND TO X_F, T_F
# Access to x_f, t_f can be given with a function factory.
# Access to model must be global since it changes along time
def net_f_uv():
        
    with tf.GradientTape(persistent=True) as tape:
        
        tape.watch(x_f)
        tape.watch(t_f)
        X_f = tf.stack([x_f, t_f], axis=1) # shape = (N_f,2)
        
        # Prediction, watched
        u, v = model(X_f)
            
        u_x = tape.gradient(u, x_f)
        v_x = tape.gradient(v, x_f)

    u_t = tape.gradient(u, t_f)
    v_t = tape.gradient(v, x_f)        
    
    u_xx = tape.gradient(u_x, x_f)
    v_xx = tape.gradient(v_x, x_f)
    
    del tape
    
    f_u = u_t + 0.5*v_xx + (u**2 + v**2)*v    
    f_v = v_t - 0.5*u_xx - (u**2 + v**2)*u   
    
    return f_u, f_v




