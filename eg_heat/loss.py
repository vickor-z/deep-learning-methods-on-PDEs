# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 13:22:48 2020

@author: vickor

SUSTech
"""
#%%
import tensorflow as tf
import numpy as np

#%% equation loss given by the PDE
class equation:
    '''
    '''
    def __init__(self, model):
        self.model = model

    def Heat(self, t, x):
        V = self.model(t, x)
        V_t = tf.gradients(V, t)[0]
        V_x = tf.gradients(V, x)[0]
        V_xx = tf.gradients(V_x, x)[0]
        L = V_t - V_xx
        L = tf.reduce_mean(tf.square(L))
        return L

#%% initial condition given by function f_i
class initial:
    def __init__(self, model, f):
        self.f = f
        self.model = model 
        
    def main(self, t, x):
        V = self.model(t,x)
        L = V - self.f(x)
        L = tf.reduce_mean(tf.square(L))
        return L
    
#%% boundary condition given by function f_b
class boundary:
    def __init__(self, model):
        self.model = model
        
    def dirichlet(self, t, x, f):
        V = self.model(t, x)
        L = V - f(x,t)
        L = tf.reduce_mean(tf.square(L))
        return L
    
    def neumann(self, t, x, f):
        V = self.model(t,x)
        V_x = tf.gradients(V, x)
        L = V_x - f(x,t)
        L = tf.reduce_mean(tf.square(L))
        return L


class loss:
    ''' Compute total loss for training.
    
    Args:
        model:      DGM model object
        t_interior: sampled time points in the interior of the function's domain
        x_interior: sampled space points in the interior of the function's domain
        t_initial: sampled time points at initial point (constant vector)
        x_initial: sampled space points at initial time
        t_boundary: sampled time points at boundary
        x_boundary: sampled space points at boundary
        f_i: initial condition
        f_b: boundary condition
    '''
    def __init__(self, model, t_interior, x_interior, t_initial, x_initial, t_boundary, x_boundary, f_i, f_b):
        self.model = model
        self.x_interior = x_interior
        self.t_interior = t_interior
        self.x_initial = x_initial 
        self.t_initial = t_initial
        self.x_boundary = x_boundary
        self.t_boundary = t_boundary
        self.f_i = f_i
        self.f_b = f_b

    def Heat(self):
        eq = equation(self.model)
        L1 = eq.Heat(self.t_interior, self.x_interior)
        L2 = initial(self.model, self.f_i).main(self.t_initial, self.x_initial)
        L3 = boundary(self.model).dirichlet(self.t_boundary, self.x_boundary, self.f_b)
        return L1, L2, L3

    
def sampler(n_interior, n_initial, n_boundary, a, b, t, T, dim):
    ''' Sample time-space points from the domain; points are sampled
        uniformly on the interior of the domain, at the initial/terminal time points
        and along the spatial boundary at different time points. 
    
    Args:
        n_interior: number of space points in the interior of the function's domain to sample 
        n_initial: number of space points at initial time to sample (initial condition)
        n_boundary: number of space points at boundary to sample
    ''' 

    # Sampler #1: domain interior
    x_interior = np.random.uniform(a, b, size=[n_interior, dim])
    t_interior = np.random.uniform(t, T, size=[n_interior, 1])


    # Sampler #2: initial condition
    x_initial = np.random.uniform(a, b, size=[n_initial, dim])
    t_initial = t * np.ones((n_initial, 1))
    
    # Sampler #3: boundary condition
    if dim == 1:
        t_boundary = np.random.uniform(t, T, size=[n_boundary, 1])
        x_boundary = np.random.choice([a,b],size=[n_boundary,1])
    else:
        x_boundary = np.random.uniform(a, b, size=[n_boundary, dim])
        x_boundary = (x_boundary>(b+a)/2)*(b-a) + a 
        index = np.arange(n_boundary)
        rindex = np.random.randint(0,dim,size=[n_boundary])
        x_boundary[index,rindex] = np.random.uniform(a,b,size=n_boundary)
        t_boundary = np.random.uniform(t, T, size=[n_boundary, 1])
        
    return t_interior, x_interior, t_initial, x_initial, t_boundary, x_boundary


