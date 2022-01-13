# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 13:22:48 2020

@author: vickor
SUSTech
"""
#%%
import tensorflow as tf
import numpy as np

#%%
class equation:
    def __init__(self, model):
        self.model = model
        
        
    def CH(self, eps, t, x):
        V = self.model(t, x)
        V_t = tf.gradients(V, t)[0]
        V_x = tf.gradients(V, x)[0]
        V_xx = tf.gradients(V_x, x)[0]
        V_xxx = tf.gradients(V_xx, x)[0]
        V_xxxx = tf.gradients(V_xxx, x)[0]
        U = V - V**3
        U_x = tf.gradients(U, x)[0]
        U_xx = tf.gradients(U_x, x)[0]
        L = V_t + eps * V_xxxx + U_xx
        L = tf.reduce_mean(tf.square(L))
        return L
    
    def Heat(self, t, x):
        V = self.model(t, x)
        V_t = tf.gradients(V, t)[0]
        V_x = tf.gradients(V, x)[0]
        V_xx = tf.gradients(V_x, x)[0]
        L = V_t - V_xx
        L = tf.reduce_mean(tf.square(L))
        return L
    
    def Heat_ND(self, t, x, dim, f):
        V = self.model(t, x)
        V_t = tf.gradients(V, t)[0]
        grad_V = tf.gradients(V, x)[0]
        Lap = 0
        for i in range(dim):
            Lap = Lap + tf.gradients(grad_V[:,i],x)[0][:,i] 
        Lap = tf.reshape(Lap, [-1,1])
        L = V_t - Lap - f(x, t)
        L = tf.reduce_mean(tf.square(L))
        return L
        
    
    def Heat_DDM(self, t, x):
        V = self.model(t, x)
        V_t = tf.gradients(V[:,0], t)[0]
        V_x = tf.gradients(V[:,0], x)[0]
        V_xx = tf.gradients(V[:,1], x)[0]
        L = []
        L.append(V_t - V_xx)
        L.append(V_x - tf.reshape(V[:,1],[-1,1]))
        ans = 0
        for i in L:
            ans = ans + tf.reduce_mean(tf.square(i))
        return ans
    
    def Heat_DDM_ND(self, t, x, dim, f):
        V = self.model(t, x)
        V_t = tf.gradients(V[:,0], t)[0]
        
        grad_V = tf.gradients(V[:,0], x)[0]
        diff_1 = grad_V - V[:, 1:dim+1]
        
        Lap = 0
        for i in range(dim):
            Lap = Lap + tf.reshape(tf.gradients(V[:,i+1], x)[0][:,i],[-1,1])   
        diff_2 = V_t - Lap - f(x,t)
        L = tf.reduce_mean(tf.square(diff_1)) + tf.reduce_mean(tf.square(diff_2))
        return L
    
    def AC_DDM_ND(self, t, x, dim, f):
        V = self.model(t, x)
        V_t = tf.gradients(V[:,0], t)[0][:,0]
        
        grad_V = tf.gradients(V[:,0], x)[0]
        diff_1 = grad_V - V[:, 1:dim+1]
        
        U = tf.reshape(V[:,0],[-1,1])
        Lap = 0
        for i in range(dim):
            Lap = Lap + tf.gradients(V[:,i+1], x)[0][:,i]   
        diff_2 = V_t - Lap - f(x,t) - U + U**3
        L = tf.reduce_mean(tf.square(diff_1)) + tf.reduce_mean(tf.square(diff_2))
        return L
    
    def forth(self, t, x, f):
        V = self.model(t, x)
        V_t = tf.gradients(V, t)[0]
        V_x = tf.gradients(V, x)[0]
        V_xx = tf.gradients(V_x, x)[0]
        V_xxx = tf.gradients(V_xx, x)[0]
        V_xxxx = tf.gradients(V_xxx, x)[0]
        diff = V_t + V_xxxx + (1 + V**2)**0.5 - f(x,t)
        return tf.reduce_mean(tf.square(diff))
    
    def kdv(self, t, x):
        V = self.model(t, x)
        V_t = tf.gradients(V, t)[0]
        V_x = tf.gradients(V, x)[0]
        V_xx = tf.gradients(V_x, x)[0]
        V_xxx = tf.gradients(V_xx, x)[0]
        diff = V_t - 6*V**2 *V_x + V_xxx
        return tf.reduce_mean(tf.square(diff))
    
    def kdv_DDM(self, t, x):
        V = self.model(t, x)
        V_t = tf.gradients(V[:,0], t)[0]
        V_x = tf.gradients(V[:,0], x)[0]
        V_xx = tf.gradients(V[:,1], x)[0]
        V_xxx = tf.gradients(V[:,2], x)[0]
        diff_1 = V_t - 6*tf.reshape(V[:,0],[-1,1])**2* tf.reshape(V[:,1],[-1,1]) + V_xxx
        diff_2 = V_x - tf.reshape(V[:,1],[-1,1])
        diff_3 = V_xx - tf.reshape(V[:,2],[-1,1])
        ans = tf.reduce_mean(tf.square(diff_1)) + tf.reduce_mean(tf.square(diff_2)) + tf.reduce_mean(tf.square(diff_3))
        return ans
        
class initial:
    def __init__(self, model, f):
        self.f = f
        self.model = model 
        
    def main(self, t, x):
        V = self.model(t,x)
        L = V - self.f(x)
        L = tf.reduce_mean(tf.square(L))
        return L
    
    def DDM(self, t, x):
        V = self.model(t,x)
        L = tf.reshape(V[:,0],[-1,1]) - self.f(x)
        L = tf.reduce_mean(tf.square(L))
        return L
    
    
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
        
    def dirichlet_DDM(self, t, x, f):
        V = self.model(t, x)
        L = tf.reshape(V[:,0],[-1,1]) - f(x,t)
        L = tf.reduce_mean(tf.square(L))
        return L
    
    def neumann_DDM(self, t, x, f):
        V = self.model(t, x)
        L = tf.reshape(V[:,1],[-1,1]) - f(x,t)
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
    
    def Heat_ND(self, f, dim, l1, l2):
        eq = equation(self.model)
        L1 = eq.Heat_ND(self.t_interior, self.x_interior, dim, f)
        L2 = l1 * initial(self.model, self.f_i).main(self.t_initial, self.x_initial)
        L3 = l2 * boundary(self.model).dirichlet(self.t_boundary, self.x_boundary, self.f_b)
        return L1, L2, L3
    
    def Heat_DDM(self):
        eq = equation(self.model)
        L1 = eq.Heat_DDM(self.t_interior, self.x_interior)
        L2 = initial(self.model, self.f_i).DDM(self.t_initial, self.x_initial)
        L3 = boundary(self.model).dirichlet_DDM(self.t_boundary, self.x_boundary, self.f_b)
        return L1, L2, L3
        
    def Heat_DDM_ND(self, f, dim, l1, l2):
        eq = equation(self.model)
        L1 = eq.Heat_DDM_ND(self.t_interior, self.x_interior, dim, f)
        L2 = l1 * initial(self.model, self.f_i).DDM(self.t_initial, self.x_initial)
        L3 = l2 * boundary(self.model).dirichlet_DDM(self.t_boundary, self.x_boundary, self.f_b)
        return L1, L2, L3
    
    def AC_DDM_ND(self, f, dim, l1, l2):
        eq = equation(self.model)
        L1 = eq.AC_DDM_ND(self.t_interior, self.x_interior, dim, f)
        L2 = l1 * initial(self.model, self.f_i).DDM(self.t_initial, self.x_initial)
        L3 = l2 * boundary(self.model).dirichlet_DDM(self.t_boundary, self.x_boundary, self.f_b)
        return L1, L2, L3
    
    def forth(self, f):
        eq = equation(self.model)
        L1 = eq.forth(self.t_interior, self.x_interior, f)
        L2 = initial(self.model, self.f_i).DDM(self.t_initial, self.x_initial)
        L3 = boundary(self.model).dirichlet_DDM(self.t_boundary, self.x_boundary, self.f_b)
        return L1, L2, L3
    
    def kdv(self):
        eq = equation(self.model)
        L1 = eq.kdv(self.t_interior, self.x_interior)
        L2 = initial(self.model, self.f_i).main(self.t_initial, self.x_initial)
        L3 = boundary(self.model).dirichlet(self.t_boundary, self.x_boundary, self.f_b)
        return L1, L2, L3
    
    def kdv_DDM(self):
        eq = equation(self.model)
        L1 = eq.kdv_DDM(self.t_interior, self.x_interior)
        L2 = initial(self.model, self.f_i).DDM(self.t_initial, self.x_initial)
        L3 = boundary(self.model).dirichlet_DDM(self.t_boundary, self.x_boundary, self.f_b)
        return L1, L2, L3
    
    def kdv_DDM_para(self, l1, l2):
        eq = equation(self.model)
        L1 = eq.kdv_DDM(self.t_interior, self.x_interior)
        L2 = l1 * initial(self.model, self.f_i).DDM(self.t_initial, self.x_initial)
        L3 = l2 * boundary(self.model).dirichlet_DDM(self.t_boundary, self.x_boundary, self.f_b)
        return L1, L2, L3
    
    def kdv_para(self, l1, l2):
        eq = equation(self.model)
        L1 = eq.kdv(self.t_interior, self.x_interior)
        L2 = l1 * initial(self.model, self.f_i).main(self.t_initial, self.x_initial)
        L3 = l2 * boundary(self.model).dirichlet(self.t_boundary, self.x_boundary, self.f_b)
        return L1, L2, L3
    
    
def sampler(n_interior, n_initial, n_boundary, a, b, t, T, dim):
    ''' Sample time-space points from the domain; points are sampled
        uniformly on the interior of the domain, at the initial/terminal time points
        and along the spatial boundary at different time points. 
    
    Args:
        n_interior: number of space points in the interior of the function's domain to sample 
        n_initial: number of space points at initial time to sample (initial condition)
    ''' 

    # Sampler #1: domain interior
    #x_interior = np.random.uniform(a, b, size=[n_interior, dim])
    x_interior = np.random.uniform(a, b, size=[n_interior, dim])
    t_interior = np.random.uniform(t, T, size=[n_interior, 1])


    # Sampler #2: initial condition
    x_initial = np.random.uniform(a, b, size=[n_initial, dim])
    t_initial = t * np.ones((n_initial, 1))
    
    if dim == 1:
        t_boundary = np.random.uniform(t, T, size=[n_boundary, 1])
        x_boundary = np.random.choice([a,b],size=[n_boundary,1])
    else:
        # Sampler #3: boundary condition
        x_boundary = np.random.uniform(a, b, size=[n_boundary, dim])
#        x_boundary = (x_boundary>(b+a)/2)*(b-a) + a 
        index = np.arange(n_boundary)
        rindex = np.random.randint(0,dim,size=[n_boundary])
        x_boundary[index,rindex] = (np.random.uniform(a,b,size=n_boundary) >(b+a)/2)*(b-a) + a
        t_boundary = np.random.uniform(t, T, size=[n_boundary, 1])
        
    return t_interior, x_interior, t_initial, x_initial, t_boundary, x_boundary

#def sampler_fd(n_interior, n_initial, n_boundary, a, b, t, T, dim):
#    x_interior = np.linspace(a,b,n_interior)
#    t_interior = np.linspace(t,T,n_interior)
