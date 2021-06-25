#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 15:37:29 2021

@author: FMagnani

"""


def plot_Adam_history(ax, Adam_loss_hist):
        
    Adam_its = len(Adam_loss_hist)
    Adam_x_axis = range(Adam_its)

    ax.plot(Adam_x_axis,Adam_loss_hist, 'r', label='Adam optimization')
    ax.legend()
    ax.set_title('Loss history for the Adam optimization')
    ax.set_ylabel('Loss value')
    ax.set_xlabel('Iteration number')

    return ax



