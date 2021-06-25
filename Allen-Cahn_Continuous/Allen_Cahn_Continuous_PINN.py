#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 16:14:24 2021

@author: FMagnani

"""

import sys
sys.path.insert(0, '../Utils/')

import tensorflow as tf

from NeuralNet import neural_net, PhysicsInformedNN

#%%

class Allen_Cahn_Continuous_PINN(PhysicsInformedNN):
    
    def __init__(self, x0, u0, x_ub, x_lb, t_b, x_f, t_f, X_star, ub, lb, layers):
        
        super(Allen_Cahn_Continuous_PINN, self).__init__()
        
        # Network architecture
        self.model = neural_net(ub, lb, layers)        
        
        # Data initialization
        self.x0 = x0
        self.t0 = 0*x0
        self.u0 = u0
        self.x_lb = x_lb
        self.x_ub = x_ub
        self.t_ub = t_b
        self.t_lb = t_b
        self.x_f = x_f
        self.t_f = t_f
        self.X_star = X_star
        
    # Loss definition
    def loss(self):
        
        x0 = self.x0
        t0 = self.t0
        u0 = self.u0
    
        # Loss from supervised learning (at t=0)
        X0 = tf.stack([x0, t0], axis=1)
        u_pred = self.model(X0)
        y0 = tf.reduce_mean(tf.square(u0 - u_pred))
    
        # Loss from PDE at the collocation pts
        f_u = self.net_f_u()
        yS = tf.reduce_mean(tf.square(f_u))
          
        # Loss from boundary conditions
        u_lb_pred, u_x_lb_pred = self.net_u(self.x_lb, self.t_lb)
        u_ub_pred, u_x_ub_pred = self.net_u(self.x_ub, self.t_ub)
    
        yB = tf.reduce_mean(tf.square(u_lb_pred - u_ub_pred)) + \
             tf.reduce_mean(tf.square(u_x_lb_pred - u_x_ub_pred))
    
        return y0 + yS + yB


    # Needed by the Loss
    def net_u(self, x, t):
    
        with tf.GradientTape() as tape:
            tape.watch(x)
            tape.watch(t)
            X = tf.stack([x, t], axis=1) # shape = (N_f,2)
        
            u = self.model(X)
         
        u_x = tape.gradient(u, x)

        return u, u_x

    # Needed by the Loss
    def net_f_u(self):
        
        x_f = self.x_f
        t_f = self.t_f
        
        with tf.GradientTape(persistent=True) as tape:    
            tape.watch(x_f)
            tape.watch(t_f)
            X_f = tf.stack([x_f, t_f], axis=1) # shape = (N_f,2)
            
            u = self.model(X_f)
                
            u_x = tape.gradient(u, x_f)
            
        u_t = tape.gradient(u, t_f)
            
        u_xx = tape.gradient(u_x, x_f)
    
        del tape
    
        f_u = u_t - 5.0*u + 5.0*u**3 - 0.0001*u_xx   
    
        return f_u

    # For the final prediction 
    # def predict(self, x, t):
                
    #     with tf.GradientTape(persistent=True) as tape:    
    #         tape.watch(x)
    #         tape.watch(t)
    #         X = tf.stack([x, t], axis=1) # shape = (N_f,2)
            
    #         u = self.model(X)
                
    #         u_x = tape.gradient(u, x)
            
    #     u_t = tape.gradient(u, t)
            
    #     u_xx = tape.gradient(u_x, x)
    
    #     del tape
    
    #     f_u = u_t - 5.0*u + 5.0*u**3 - 0.0001*u_xx     
    
    #     return u, f_u


