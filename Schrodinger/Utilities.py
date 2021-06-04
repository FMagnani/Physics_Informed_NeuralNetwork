#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 20:05:23 2021

@author: FMagnani

"""

import tensorflow as tf


#%%

def net_uv(NN, x, t):
    
    X = tf.concat([x,t],1)
    X = tf.Variable(X)
    
    with tf.GradientTape(persistent=True) as tape:
        u, v = NN(X)
         
    u_X = tape.gradient(u, X)
    v_X = tape.gradient(v, X)

    u_x = u_X[:,0:1]
    v_x = v_X[:,1:2]

    return u, v, u_x, v_x

def net_f_uv(NN, x, t):
    """
   ARGS:
        NN: The Neural Network used as solution. An instance of neural_net class
        x, t: Input 
    RETURNS:
        f_u, f_v: output of the NN    
    Used to compute the loss referring to the Shrodinger eq condition.
    It computes the first time derivatives and the second space derivatives
    of the solution, represented by 'neural_net'.
    """
    
    X = tf.concat([x,t],1)
    X = tf.Variable(X) 
    
    with tf.GradientTape(persistent=True) as tape_1:
        with tf.GradientTape(persistent=True) as tape_2:
            u, v = NN(X)
            
            u_X = tape_2.gradient(u, X)
            v_X = tape_2.gradient(v, X)

    u_t = u_X[:,1]        
    v_t = v_X[:,1]        
    
    u_xx = tape_1.gradient(u_X, X)[:,0]
    v_xx = tape_1.gradient(v_X, X)[:,0]
    
    u_t = tf.cast(u_t, dtype='float32')
    v_t = tf.cast(u_t, dtype='float32')
    u_xx = tf.cast(u_xx, dtype='float32')
    v_xx = tf.cast(v_xx, dtype='float32')
    
    f_u = u_t + 0.5*v_xx + (u**2 + v**2)*v    
    f_v = v_t - 0.5*u_xx - (u**2 + v**2)*u   
    
    return f_u, f_v




