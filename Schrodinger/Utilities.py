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
    
def train_step(x0,t0, u0,v0, n_iterations, optimizer):
        
    with tf.GradientTape() as tape:
        
        loss_value = loss(x0,t0, u0,v0)
 
    grads = tape.gradient(loss_value, model.trainable_variables)
    
    optimizer.apply_gradients(zip(grads, model.trainable_variables))


def loss(x0,t0, u0,v0):
    
    # Loss from supervised learning (at t=0)
    X0 = tf.stack([x0, t0], axis=1)
    u_pred, v_pred = model(X0)
    y0 = tf.reduce_mean(tf.square(u0 - u_pred)) + \
         tf.reduce_mean(tf.square(v0 - v_pred))
    
    # Loss from Schrodinger constraint (at the anchor pts)
    f_u, f_v = net_f_uv()
    yS = tf.reduce_mean(tf.square(f_u)) + \
          tf.reduce_mean(tf.square(f_v))
          
    # Loss from boundary conditions
    u_lb_pred, v_lb_pred, u_x_lb_pred, v_x_lb_pred = net_uv(x_lb, t_lb)
    u_ub_pred, v_ub_pred, u_x_ub_pred, v_x_ub_pred = net_uv(x_ub, t_ub)
    
    yB = tf.reduce_mean(tf.square(u_lb_pred - u_ub_pred)) + \
         tf.reduce_mean(tf.square(v_lb_pred - v_ub_pred)) + \
         tf.reduce_mean(tf.square(u_x_lb_pred - u_x_ub_pred)) + \
         tf.reduce_mean(tf.square(v_x_lb_pred - v_x_ub_pred))
    

    return y0 + yS + yB



def net_uv(x, t):
    
    with tf.GradientTape(persistent=True) as tape:
        
        tape.watch(x)
        tape.watch(t)
        X = tf.stack([x, t], axis=1) # shape = (N_f,2)
        
        u, v = model(X)
         
    u_x = tape.gradient(u, x)
    v_x = tape.gradient(v, x)

    return u, v, u_x, v_x



def create_net_f_uv(x_f, t_f):
    
    # NEEDS ACCESS TO MODEL
    def net_f_uv():
        
        with tf.GradientTape(persistent=True) as tape:
            
            tape.watch(x_f)
            tape.watch(t_f)
            X_f = tf.stack([x_f, t_f], axis=1) # shape = (N_f,2)
            
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

    return net_f_uv










