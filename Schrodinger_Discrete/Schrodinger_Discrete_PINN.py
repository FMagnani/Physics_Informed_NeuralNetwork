#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 12:02:16 2021

@author: FMagnani

"""

import sys
sys.path.insert(0, '../Utils/')

import tensorflow as tf
import numpy as np

from NeuralNet import neural_net, PhysicsInformedNN

#%%

class neural_net_2out(neural_net):
    """
    Input: tensor of shape (N,1) <-- The x coordinates
    Output: tensor of shape (N,q+1)
    
    The 2 tensors in output will be the RK stages for the Real and for the Imaginary part.
    """
    
    def __init__(self, ub, lb, layers):
        super(neural_net_2out, self).__init__(ub, lb, layers)

    def call(self, inputs, training=False):
        """
        Input will have shape: (N,1) <-- the x coordinates
        Output will be 2 tensors with shape: (N,q+1)
        
        The input is duplicated before to be fed into the net.
        """
        
#        doubled_inputs = tf.stack([inputs,inputs], axis=0)
    
        output = super(neural_net_2out, self).call(inputs)
        
        half_len = int(output.shape[1]/2)
        
        U = output[:, :half_len]
        V = output[:, half_len:]
        
        return U, V


class Schrodinger_PINN(PhysicsInformedNN):
    
    def __init__(self, x0, u0,v0, layers, dt, lb, ub, q):
        
        super(Schrodinger_PINN, self).__init__()
        
        # Network architecture
        self.model = neural_net_2out(ub, lb, layers)        
        
        # Data initialization
        self.x0 = x0
        x1 = np.vstack((lb,ub))
        self.x1 = tf.convert_to_tensor(x1)
        self.u0 = u0
        self.v0 = v0
        
        self.dt = dt
        self.q = max(q,1)
        
        # Load IRK weights
        tmp = np.float64(np.loadtxt('../Utils/IRK_weights/Butcher_IRK%d.txt' % (q), ndmin = 2))
        self.IRK_weights = np.reshape(tmp[0:q**2+q], (q+1,q))
        self.IRK_times = tmp[q**2+q:]
 
    
    def loss(self):
        
        u0 = self.u0
        v0 = self.v0
        x0 = self.x0
        x1 = self.x1
        
        U0_pred, V0_pred = self.net_UV0(x0) # N x (q+1), N x (q+1)
        U1_pred, V1_pred, U1_x_pred, V1_x_pred = self.net_UV1(x1) # N1 x (q+1)
        
        y = tf.reduce_sum(tf.square(u0 - U0_pred)) + \
            tf.reduce_sum(tf.square(v0 - V0_pred))
        
        yB = tf.reduce_sum(tf.square(U1_pred[0,:] - U1_pred[1,:])) + \
            tf.reduce_sum(tf.square(U1_x_pred[0,:] - U1_x_pred[1,:])) + \
            tf.reduce_sum(tf.square(V1_pred[0,:] - V1_pred[1,:])) + \
            tf.reduce_sum(tf.square(V1_x_pred[0,:] - V1_x_pred[1,:])) 
    
        return y + yB


    def net_UV0(self, x):
        
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)

            U1, V1 = self.model(x)
            U = U1[:,:-1]
            V = V1[:,:-1]
            
            U_x = tape.gradient(U, x)
            V_x = tape.gradient(V, x)
        
        U_xx = tape.gradient(U_x, x)
        V_xx = tape.gradient(V_x, x)
        
        del tape
                
        F_u = -0.5*V_xx - (U**2 + V**2)*V
        F_v = -0.5*U_xx + (U**2 + V**2)*U
            
        U0 = U1 - self.dt*tf.matmul(F_u, self.IRK_weights.T)
        V0 = U1 - self.dt*tf.matmul(F_v, self.IRK_weights.T)
        
        return U0, V0


    def net_UV1(self, x):
        
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            U1, V1 = self.model(x)

        U1_x = tape.gradient(U1, x)  
        V1_x = tape.gradient(V1, x)

        return U1,V1, U1_x,V1_x # N x (q+1)


    def predict(self, x):
        
        U1, V1 = self.model(x)
        
        return U1, V1















