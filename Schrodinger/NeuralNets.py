#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 18:41:40 2021

@author: FMagnani

"""

import tensorflow as tf
from tensorflow.keras.layers import Dense


#%%

class neural_net(tf.keras.Model):
    
    def __init__(self, ub, lb, hidden_dim=100):
        super(neural_net, self).__init__()
        
        self.lb = lb
        self.ub = ub

        self.hidden_dim = hidden_dim
        self.out_dim = 2
        
        self.hidden_1 = Dense(self.hidden_dim, activation='tanh',
                    bias_initializer="zeros")

        self.hidden_2 = Dense(self.hidden_dim, activation='tanh',
                    bias_initializer="zeros")

        self.hidden_3 = Dense(self.hidden_dim, activation='tanh',
                    bias_initializer="zeros")

        self.hidden_4 = Dense(self.hidden_dim, activation='tanh',
                    bias_initializer="zeros")

        self.last_layer = Dense(self.out_dim,
                    bias_initializer="zeros",
                    dtype = 'float64')

    def call(self, X):
        
        H = 2.0*(X - self.lb)/(self.ub - self.lb) - 1.0

        H = self.hidden_1(H)
        H = self.hidden_2(H)
        H = self.hidden_3(H)
        H = self.hidden_4(H)
        
        H = self.last_layer(H)
    
        u = H[:,0]
        v = H[:,1]
    
        return u, v



class Schrodinger_PINN():
    
    def __init__(self, x0, u0, v0, x_ub, x_lb, t_b, x_f, t_f, X_star, ub, lb, hidden_dim=100):
        
        self.model = neural_net(ub, lb, hidden_dim)        
        
        self.x0 = x0
        self.t0 = 0*x0
        self.u0 = u0
        self.v0 = v0
        self.x_lb = x_lb
        self.x_ub = x_ub
        self.t_ub = t_b
        self.t_lb = t_b
        self.x_f = x_f
        self.t_f = t_f
        self.X_star = X_star
        
        
    def train_step(self, n_iterations, optimizer):
        
        with tf.GradientTape() as tape:
            loss_value = self.loss(self.x0,self.t0, self.u0,self.v0)
 
        grads = tape.gradient(loss_value, self.model.trainable_variables)    
        optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
  
        
    def loss(self, x0,t0, u0,v0):
    
        # Loss from supervised learning (at t=0)
        X0 = tf.stack([x0, t0], axis=1)
        u_pred, v_pred = self.model(X0)
        y0 = tf.reduce_mean(tf.square(u0 - u_pred)) + \
             tf.reduce_mean(tf.square(v0 - v_pred))
    
        # Loss from Schrodinger constraint (at the anchor pts)
        f_u, f_v = self.net_f_uv()
        yS = tf.reduce_mean(tf.square(f_u)) + \
              tf.reduce_mean(tf.square(f_v))
          
        # Loss from boundary conditions
        u_lb_pred, v_lb_pred, u_x_lb_pred, v_x_lb_pred = self.net_uv(self.x_lb, self.t_lb)
        u_ub_pred, v_ub_pred, u_x_ub_pred, v_x_ub_pred = self.net_uv(self.x_ub, self.t_ub)
    
        yB = tf.reduce_mean(tf.square(u_lb_pred - u_ub_pred)) + \
             tf.reduce_mean(tf.square(v_lb_pred - v_ub_pred)) + \
             tf.reduce_mean(tf.square(u_x_lb_pred - u_x_ub_pred)) + \
             tf.reduce_mean(tf.square(v_x_lb_pred - v_x_ub_pred))
    
        return y0 + yS + yB


    def net_uv(self, x, t):
    
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            tape.watch(t)
            X = tf.stack([x, t], axis=1) # shape = (N_f,2)
        
            u, v = self.model(X)
         
        u_x = tape.gradient(u, x)
        v_x = tape.gradient(v, x)

        return u, v, u_x, v_x

   
    def net_f_uv(self):
        
        x_f = self.x_f
        t_f = self.t_f
        
        with tf.GradientTape(persistent=True) as tape:    
            tape.watch(x_f)
            tape.watch(t_f)
            X_f = tf.stack([x_f, t_f], axis=1) # shape = (N_f,2)
            
            u, v = self.model(X_f)
                
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

     
    def predict(self, x, t):
                
        with tf.GradientTape(persistent=True) as tape:    
            tape.watch(x)
            tape.watch(t)
            X = tf.stack([x, t], axis=1) # shape = (N_f,2)
            
            u, v = self.model(X)
                
            u_x = tape.gradient(u, x)
            v_x = tape.gradient(v, x)
            
        u_t = tape.gradient(u, t)
        v_t = tape.gradient(v, t)        
            
        u_xx = tape.gradient(u_x, x)
        v_xx = tape.gradient(v_x, x)
    
        del tape
    
        f_u = u_t + 0.5*v_xx + (u**2 + v**2)*v    
        f_v = v_t - 0.5*u_xx - (u**2 + v**2)*u   
    
        return u, v, f_u, f_v



