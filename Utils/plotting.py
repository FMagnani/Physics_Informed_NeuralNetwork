#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 15:37:29 2021

@author: FMagnani

"""


def plot_loss_history(ax, Adam_loss_hist, lbfgs_loss_hist):
        
    Adam_its = len(Adam_loss_hist)
    lbfgs_its = len(lbfgs_loss_hist)
    Adam_x_axis = range(Adam_its)
    lbfgs_x_axis = range(Adam_its, Adam_its+lbfgs_its)

    ax.plot(Adam_x_axis,Adam_loss_hist, 'r', label='Adam optimization')
    ax.plot(lbfgs_x_axis,lbfgs_loss_hist, 'b', label='LBFGS optimization')
    ax.legend()
    ax.set_title('Loss history')
    ax.set_ylabel('Loss value')
    ax.set_xlabel('Iteration number')

    return ax



