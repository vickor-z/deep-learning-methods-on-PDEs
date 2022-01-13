# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 18:45:31 2019

@author: vickor
"""

# DGM, LDGM based solver for CH equation

#%% import packages
import net
import tensorflow as tf
import numpy as np
import scipy.stats as spstats
import matplotlib.pyplot as plt
import math 
import pandas as pd
import csv
import time
import matplotlib
#%% Parameters for CH
#EPS = [0.5,0.1,0.05,0.01]
eps = 0.02
fig_flag = 1
alpha = 0.5
pi = 4 * np.arctan(1.0)
mult = 1
nu = 0.05

# Solution domain
t, T = 0, 1
dt = 0.1
a, b = 0, 2 * pi
dim = 1


# Neural network parameters
num_layers = 3
nodes_per_layer = 50
learning_rate = 0.001

# Training and Sampling parameters
sampling_stages = 5000
steps_per_sample = 5
n_interior = 200
n_initial = 50
n_boundary = 50

plt.figure()
plt.figure(figsize = (18,12))

w2 = 100
w3 = 100

#%% True solution
def ue(x, t):
    ans = np.exp(-t) * np.sin(x)
    return ans

def u_initial(x):
    ans = tf.cos(x)
    return ans

#%% Sampling function
def sampler(n_interior, n_initial, n_boundary):
    ''' Sample time-space points from the domain; points are sampled
        uniformly on the interior of the domain, at the initial/terminal time points
        and along the spatial boundary at different time points. 
    
    Args:
        n_interior: number of space points in the interior of the function's domain to sample 
        n_initial: number of space points at initial time to sample (initial condition)
    ''' 

    # Sampler #1: domain interior
    #x_interior = np.random.uniform(a, b, size=[n_interior, dim])
    x_interior = np.linspace(a, b, n_interior)
    x_interior = x_interior.reshape(-1,1)
    t_interior = np.random.uniform(t, T, size=[n_interior, 1])

    # Sampler #2: initial condition
    x_initial = np.random.uniform(a, b, size=[n_initial, dim])
    t_initial = t * np.ones((n_initial, 1))

    # Sampler #3: boundary condition
    x_boundary_L = a * np.ones((n_boundary, 1))
    x_boundary_R = b * np.ones((n_boundary, 1))
    t_boundary = np.random.uniform(t, T, size=[n_boundary, 1])
    
    
    return t_interior, x_interior, t_initial, x_initial, t_boundary, x_boundary_L, x_boundary_R

#%% Loss function for Allen_Cahn equation

#%% Set up network
model = net.DeepNet(nodes_per_layer, num_layers, dim,final_trans='tanh',active = 'tanh')
model_B = net.DeepNet_4V(nodes_per_layer, num_layers, dim,final_trans='tanh')

# tensor placeholder (_tnsr suffix indicates tensors)
t_interior_tnsr = tf.placeholder(tf.float32, [None, 1])
x_interior_tnsr = tf.placeholder(tf.float32, [None,1])
t_initial_tnsr = tf.placeholder(tf.float32, [None,1])
x_initial_tnsr = tf.placeholder(tf.float32, [None,1])
t_boundary_tnsr = tf.placeholder(tf.float32, [None,1])
x_boundary_L_tnsr = tf.placeholder(tf.float32, [None,1])
x_boundary_R_tnsr = tf.placeholder(tf.float32, [None,1])

t_interior_tnsr_B = tf.placeholder(tf.float32, [None, 1])
x_interior_tnsr_B = tf.placeholder(tf.float32, [None,1])
t_initial_tnsr_B = tf.placeholder(tf.float32, [None,1])
x_initial_tnsr_B = tf.placeholder(tf.float32, [None,1])
t_boundary_tnsr_B = tf.placeholder(tf.float32, [None,1])
x_boundary_L_tnsr_B = tf.placeholder(tf.float32, [None,1])
x_boundary_R_tnsr_B = tf.placeholder(tf.float32, [None,1])
    # Solution
V = model(t_interior_tnsr, x_interior_tnsr)
V_B = model_B(t_interior_tnsr_B, x_interior_tnsr_B)


def loss(model, t_interior, x_interior, t_initial, x_initial, t_boundary, x_boundary_L, x_boundary_R):
    ''' Compute total loss for training.
    
    Args:
        model:      LDGM model object
        t_interior: sampled time points in the interior of the function's domain
        x_interior: sampled space points in the interior of the function's domain
        t_initial: sampled time points at initial point (constant vector)
        x_initial: sampled space points at initial time
    ''' 
    # Loss term #1: PDE
    # compute the PDE loss at sampled points
    
    V = model(t_interior, x_interior)
    V_t = tf.gradients(V, t_interior)[0]
    V_s = tf.gradients(V, x_interior)[0]
    V_ss = tf.gradients(V_s, x_interior)[0]
    V_sss = tf.gradients(V_ss, x_interior)[0]
    V_ssss = tf.gradients(V_sss, x_interior)[0]
    W = V - V**3
    W_s = tf.gradients(W, x_interior)[0]
    W_ss = tf.gradients(W_s, x_interior)[0]
    diff_V = V_t + eps*V_ssss + W_ss

    L1 = tf.reduce_mean(tf.square(diff_V)) 

    # Loss term #2: Initial Boundary Condition
    V = model(t_initial, x_initial)
    L2 = tf.reduce_mean(tf.square(V - u_initial(x_initial)))
    
    # Loss term #3: Boundary Condition
    VL = model(t_boundary, x_boundary_L)
    VR = model(t_boundary, x_boundary_R)
    VL_s = tf.gradients(VL, x_boundary_L)[0]
    VL_ss = tf.gradients(VL_s, x_boundary_L)[0]
    VL_sss = tf.gradients(VL_ss, x_boundary_L)[0]
    VR_s = tf.gradients(VR, x_boundary_R)[0]
    VR_ss = tf.gradients(VR_s, x_boundary_R)[0]
    VR_sss = tf.gradients(VR_ss, x_boundary_R)[0]
    L3 = tf.reduce_mean(tf.square(VL_s)+tf.square(VR_s))
    L3 = L3 + tf.reduce_mean(tf.square(VL_sss)+tf.square(VR_sss))
    return L1, w2*L2, w3*L3

def loss_B(model, t_interior, x_interior, t_initial, x_initial, t_boundary, x_boundary_L, x_boundary_R):
    ''' Compute total loss for training.
    
    Args:
        model:      DGM model object
        t_interior: sampled time points in the interior of the function's domain
        x_interior: sampled space points in the interior of the function's domain
        t_initial: sampled time points at initial point (constant vector)
        x_initial: sampled space points at initial time
    ''' 
    # Loss term #1: PDE
    # compute the PDE loss at sampled points
    # Numerical integration for the time fractional derivative with Gauss-Jacobi quadrature.
    V = model(t_interior,x_interior)
    V_t = tf.gradients(V[:,0],t_interior)[0]
    V_x = tf.gradients(V[:,0],x_interior)[0]
    V_xx = tf.gradients(V[:,1],x_interior)[0]
    
    U_x = tf.gradients(V[:,2],x_interior)[0]
    U_xx = tf.gradients(V[:,3],x_interior)[0]
    
    
    
    diff_V1 = V_t - U_xx
    diff_V2 = tf.reshape(V[:,1],[-1,1]) - V_x
    diff_V3 = tf.reshape(V[:,2],[-1,1]) + eps * V_xx + tf.reshape(V[:,0],[-1,1]) - tf.reshape(V[:,0],[-1,1])**3
    diff_V4 = tf.reshape(V[:,3],[-1,1]) - U_x

    L1 = 0
    diff_V = []
    diff_V.append(diff_V1)
    diff_V.append(diff_V2)
    diff_V.append(diff_V3)
    diff_V.append(diff_V4)
    for i in diff_V:
        L1 = L1 + tf.reduce_mean(tf.square(i))
    
    # Loss term #2: Initial Boundary Condition
    V = model(t_initial, x_initial)
    V = tf.reshape(V[:,0], [-1,1])
    L2 = tf.reduce_mean(tf.square(V - u_initial(x_initial)))
    
    # Loss term #3: Boundary Condition
    VL = model(t_boundary, x_boundary_L)
    VR = model(t_boundary, x_boundary_R)
    L3 = tf.reduce_mean(tf.square(VL[:,1])+tf.square(VR[:,1]))
    L3 = L3+tf.reduce_mean(tf.square(VL[:,3])+tf.square(VR[:,3]))

    return L1, w2*L2, w3*L3


#%% loss
L1_tnsr, L2_tnsr, L3_tnsr = loss(model, t_interior_tnsr, x_interior_tnsr, t_initial_tnsr, x_initial_tnsr, t_boundary_tnsr, x_boundary_L_tnsr, x_boundary_R_tnsr)
loss_tnsr = L1_tnsr + L2_tnsr + L3_tnsr

L1_tnsr_B, L2_tnsr_B, L3_tnsr_B = loss_B(model_B, t_interior_tnsr_B, x_interior_tnsr_B, t_initial_tnsr_B, 
                                       x_initial_tnsr_B, t_boundary_tnsr_B, x_boundary_L_tnsr_B, x_boundary_R_tnsr_B)
loss_tnsr_B = L1_tnsr_B + L2_tnsr_B + L3_tnsr_B




# optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_tnsr)
optimizer_B = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_tnsr_B)

# initialize variables
np.random.seed(0)
init_op = tf.global_variables_initializer()
    
# open session
sess = tf.Session()
sess.run(init_op)

#%% prepration
N = 100
t = 0.0
T = 1.0

x = np.linspace(a,b,N+1)
t = np.linspace(t,T,N+1)
XX, TT = np.meshgrid(x,t)
#plt.subplot(2,3,1)
XX = np.reshape(XX,[-1,1])
TT = np.reshape(TT,[-1,1])

t = 0.0
T = 1.0


#%% Train network

# record the error in every plot
er_record_DGM = []
er_record_Force = []
cputime_record_DGM = []
N = 100 
h = (b-a)/N
x_plot = (np.linspace(a, b, N)).reshape(-1,1)
t_plot = T * np.ones_like(x_plot)
exact_sol = ue(x_plot, T)
step = 0
delta_t = 0
k=1
plt.subplots_adjust(hspace=1, wspace=0.3)
DGM_cputime = 0
Force_cputime = 0
DGM_cputime_record = []
Force_cputime_record = []

for i in range(sampling_stages):
    # sample first
    
    t_interior, x_interior, t_initial, x_initial, t_boundary, x_boundary_L, x_boundary_R = sampler(n_interior, n_initial, n_boundary)

    # for each sample, take required steps with optimizer
    
    for _ in range(steps_per_sample):
        
        t1 = time.time()
        loss, L1, L2, L3, _ = sess.run([loss_tnsr, L1_tnsr, L2_tnsr, L3_tnsr, optimizer],
                                    feed_dict = {t_interior_tnsr:t_interior, 
                                                x_interior_tnsr:x_interior, 
                                                t_initial_tnsr:t_initial, 
                                                x_initial_tnsr:x_initial,
                                                t_boundary_tnsr:t_boundary,
                                                x_boundary_L_tnsr:x_boundary_L,
                                                x_boundary_R_tnsr:x_boundary_R,})
        t2 = time.time()
        DGM_cputime = DGM_cputime + t2-t1
        loss_B, L1_B, L2_B, L3_B, _ = sess.run([loss_tnsr_B, L1_tnsr_B, L2_tnsr_B, L3_tnsr_B, optimizer_B],
                                    feed_dict = {t_interior_tnsr_B:t_interior, 
                                                x_interior_tnsr_B:x_interior, 
                                                t_initial_tnsr_B:t_initial, 
                                                x_initial_tnsr_B:x_initial,
                                                t_boundary_tnsr_B:t_boundary,
                                                x_boundary_L_tnsr_B:x_boundary_L,
                                                x_boundary_R_tnsr_B:x_boundary_R,})
        t3 = time.time()
        Force_cputime = Force_cputime + t3-t2
        
        step = step + 1
        
        if step%1000 == 0:
             k = k+1
             DGM_cputime_record.append(DGM_cputime)
             Force_cputime_record.append(Force_cputime)
             print(loss, L1, L2, L3, loss_B,L1_B,L2_B,L3_B, step)

print(DGM_cputime,Force_cputime)
#%% numerical spectral method
N = 64
dt = 0.01
t = 0
T = 1
x = np.linspace(a, b, 2 * N + 1)
u0 = np.cos(x)
w = u0 - u0**3;


i = np.linspace(0, 2*N, 2*N+1)
W1 = np.exp(-2 * pi * N / (2 * N + 1) * i * 1j) 
W2 = np.exp(2 * pi * N / (2 * N + 1) * i * 1j)   

i = np.linspace(-N, N, 2*N+1);
i = i ** 2;
A = (1 + dt * eps * i**2)

hatu = np.fft.ifft(u0 * W1)

up = u0
step = 0

numerical_record = []
numerical_record.append(u0)
t1 = time.time()
while t < T:
    hatw = np.fft.ifft(w * W1);
    
    hatu = (hatu  + dt * i  * hatw)/A;
    
    u = np.real(np.fft.fft(hatu) * W2)
    w = u - u**3
    t = t + dt
    
    
    numerical_record.append(u)
t2 = time.time()
FD_time = t2 - t1
#%% plot the training process 
plt.figure(figsize=(16,3))
MAP = plt.cm.Spectral
norm = matplotlib.colors.Normalize(vmin=-1, vmax=1)
# time nodes for plot
x_plot = np.linspace(a,b,2*N+1)
t_plot = np.linspace(0,T,int(T/dt)+1)
XX,TT = np.meshgrid(x_plot,t_plot)
fig_flag= 1
ftsize = 'xx-large'
test_input = [[[XX[i][j],TT[i][j]] for i in range(101)] for j in range(2*N+1)]
nodes = []
for i in test_input:
    nodes += i
nodes = np.array(nodes)
predict_sol = sess.run([V], feed_dict = {t_interior_tnsr:nodes[:,1].reshape(-1,1), x_interior_tnsr:nodes[:,0].reshape(-1,1)})
DDM_sol = sess.run([V_B], feed_dict = {t_interior_tnsr_B:nodes[:,1].reshape(-1,1), x_interior_tnsr_B:nodes[:,0].reshape(-1,1)})

plt.subplot(1,3,fig_flag)
plt.tick_params(labelsize='large')
plt.contourf(XX,TT,np.array(numerical_record),cmap=MAP, norm=norm)
plt.title('reference solution', fontsize=ftsize)
plt.colorbar()
plt.ylabel('$\epsilon$=%s'%(eps), fontsize=ftsize)
#plt.ylabel('$\epsilon$=0.03', fontsize=ftsize)

plt.subplot(1,3,fig_flag+1)
plt.tick_params(labelsize='large')
fig2 = plt.contourf(XX,TT,predict_sol[0].reshape(2*N+1,101).T,cmap=MAP,norm=norm)
plt.title('DGM solution', fontsize=ftsize)
plt.colorbar()

plt.subplot(1,3,fig_flag+2)
plt.tick_params(labelsize='large')
plt.contourf(XX,TT,DDM_sol[0][:,0].reshape(2*N+1,101).T,cmap=MAP, norm=norm)
plt.title('LDGM solution', fontsize=ftsize)
plt.colorbar()
fig_flag = fig_flag + 3
#plt.savefig('figure/eps='%(eps),format='eps')
#%%
u1 = np.array(numerical_record)
u2 = predict_sol[0].reshape(2*N+1,101).T
u3 = DDM_sol[0][:,0].reshape(2*N+1,101).T

er1 = np.linalg.norm(u1-u2)/np.linalg.norm(u1)
er2 = np.linalg.norm(u1-u3)/np.linalg.norm(u1)
print(er1,er2)