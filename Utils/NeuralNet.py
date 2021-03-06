#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 19 16:36:49 2021

@author: FMagnani

"""

import tensorflow as tf
import scipy.optimize

import time
from tqdm import tqdm
import numpy as np

#%%

class neural_net(tf.keras.Sequential):
    
    def __init__(self, ub, lb, layers):
        super(neural_net, self).__init__()
        
#        layers is something like [2, 100, 100, 100, 100, 2]
       
        tf.keras.backend.set_floatx('float64')
        
        self.t_last_callback = 0
        
        self.lb = lb
        self.ub = ub

        self.add(tf.keras.layers.InputLayer(input_shape=(layers[0],)))

        self.add(tf.keras.layers.Lambda(
                        lambda X: 2.0*(X-self.lb)/(self.ub-self.lb)-1.0))

        for width in layers[1:-1]:
            self.add(tf.keras.layers.Dense(
                width, activation=tf.nn.tanh,
                kernel_initializer="glorot_normal"))
        
        self.add(tf.keras.layers.Dense(
                layers[-1], activation=None,
                kernel_initializer="glorot_normal"))

        
        # Computing the sizes of weights/biases for future decomposition
        self.sizes_w = []
        self.sizes_b = []
        for i, width in enumerate(layers):
            if i != 1:
                self.sizes_w.append(int(width * layers[1]))
                self.sizes_b.append(int(width if i != 0 else layers[1]))


    def get_weights(self, convert_to_tensor=True):
        w = []
        for layer in self.layers[1:]:
            weights_biases = layer.get_weights()
            weights = weights_biases[0].flatten()
            biases = weights_biases[1]
            w.extend(weights)
            w.extend(biases)
        if convert_to_tensor:
            w = tf.convert_to_tensor(w)
        
        # Make the output Fortran contiguous
        w = np.copy(w, order='F')
        
        return w


    def set_weights(self, w):
        for i, layer in enumerate(self.layers[1:]):
            start_weights = sum(self.sizes_w[:i]) + sum(self.sizes_b[:i])
            end_weights = sum(self.sizes_w[:i+1]) + sum(self.sizes_b[:i])
            weights = w[start_weights:end_weights]
            w_div = int(self.sizes_w[i] / self.sizes_b[i])
            weights = tf.reshape(weights, [w_div, self.sizes_b[i]])
            biases = w[end_weights:end_weights + self.sizes_b[i]]
            weights_biases = [weights, biases]
            layer.set_weights(weights_biases)


class PhysicsInformedNN():
    
    def __init__(self):
        pass
   
    def loss_and_flat_grad(self, w):
        
        with tf.GradientTape() as tape:
            self.model.set_weights(w)
            loss_value = self.loss()
        grad = tape.gradient(loss_value, self.model.trainable_variables)
        grad_flat = []
        for g in grad:
            grad_flat.append(tf.reshape(g, [-1]))
        grad_flat = tf.concat(grad_flat, 0)
        
        # Make the output Fortran contiguous
        loss_value = np.copy(loss_value, order='F')
        grad_flat = np.copy(grad_flat, order='F')
        
        return loss_value, grad_flat

    def loss(self):
        pass
    
    
    def train(self, Adam_iterations, LBFGS_max_iterations):
        
        # ADAM training
        Adam_hist = []
        if (Adam_iterations):
            
            print('~~ Adam optimization ~~')
            
            optimizer=tf.keras.optimizers.Adam()
            
            start_time = time.time()   
            iteration_start_time = start_time
            #Train step
            for i in range(Adam_iterations):
                current_loss = self.Adam_train_step(optimizer)
                Adam_hist.append(current_loss)
                iteration_time = str(time.time()-iteration_start_time)[:5]
                print('Loss:', current_loss.numpy(), 'time:', iteration_time, 'iter: '+str(i)+'/'+str(Adam_iterations) )                
                iteration_start_time = time.time()
            elapsed = time.time() - start_time      
            Adam_hist.append(self.loss())
            print('Training time: %.4f' % (elapsed))
            
        
        # LBFGS trainig
        self.LBFGS_hist = [self.loss()]
        if (LBFGS_max_iterations):
            
            print('~~ L-BFGS optimization ~~')
                        
            maxiter = LBFGS_max_iterations
            self.t_last_callback = time.time()
            
            results = scipy.optimize.minimize(self.loss_and_flat_grad,
                                              self.model.get_weights(),
                                              method='L-BFGS-B',
                                              jac=True,
                                              callback=self.callback,
                                              options = {'maxiter': maxiter,
                                                         'maxfun': 50000,
                                                         'maxcor': 50,
                                                         'maxls': 50,
                                                         'ftol' : 1.0 * np.finfo(float).eps})
            
            optimal_w = results.x 
            self.model.set_weights(optimal_w)
    
            print('~~ model trained ~~','\n','Final loss:',self.loss().numpy())
    
        return Adam_hist, self.LBFGS_hist
    
    @tf.function
    def Adam_train_step(self, optimizer):
        
        with tf.GradientTape() as tape:
            loss_value = self.loss()
 
        grads = tape.gradient(loss_value, self.model.trainable_variables)    
        optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
  
        return loss_value

    def callback(self, pars):
        t_interval = str(time.time()-self.t_last_callback)[:5]
        loss_value = self.loss().numpy()
        self.LBFGS_hist.append(loss_value)
        
        print('Loss:', loss_value, 'time:', t_interval)
        self.t_last_callback = time.time()

