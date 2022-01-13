# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 18:45:31 2019

@author: vickor
"""

# DGM, LDGM, DRM based solver for Possion's equation

#%% import packages
import net
import tensorflow as tf
import numpy as np
import scipy.stats as spstats
import matplotlib.pyplot as plt
import math 
import pandas as pd
import csv
import matplotlib

#%% Parameters for Allen-Cahn
eps = 0.005
alpha = 0.5
pi = 4 * np.arctan(1.0)
mult = 1
nu = 0.05

# Solution domain
t, T = 0.0, 2*pi
dt = 0.001
a, b = 0.0, 2*pi
dim = 2

# Neural network parameters
num_layers = 3
nodes_per_layer = 50
learning_rate = 0.001

# Training and Sampling parameters
sampling_stages = 1000
steps_per_sample = 5
n_interior = 200
n_initial = 50
n_boundary = 50


#%% True solution, initial condition and boundary condition
def ue(x, y):
    ans = np.sin(x)*np.sin(y)
    return ans

def f(x, y):
    ans = 2 * tf.sin(x) * tf.sin(y)
    return ans

def u_boundary(x,y):
    return tf.constant(0.0)
    
#%% sampler
def sampler(n_interior, n_boundary):
    x_interior = np.random.uniform(a, b, size=[n_interior, 1])
    y_interior = np.random.uniform(t, T, size=[n_interior, 1])

    x_boundary = np.random.uniform(a, b, size=[n_boundary, 2])
    x_boundary = (x_boundary>(b+a)/2)*(b-a) + a 
    index = np.arange(n_boundary)
    rindex = np.random.randint(0,2,size=[n_boundary])
    x_boundary[index,rindex] = np.random.uniform(a,b,size=n_boundary)
    
    y_boundary = np.reshape(x_boundary[:,1],[-1,1])
    x_boundary = np.reshape(x_boundary[:,0],[-1,1])
    return x_interior, y_interior, x_boundary, y_boundary

def loss_DGM(model, x_interior, y_interior, x_boundary, y_boundary):
    V = model(y_interior,x_interior)
    V_y = tf.gradients(V, y_interior)[0]
    V_yy = tf.gradients(V_y, y_interior)[0]
    V_x = tf.gradients(V, x_interior)[0]
    V_xx = tf.gradients(V_x, x_interior)[0]
    
    diff = V_xx + V_yy + f(x_interior,y_interior)
    L1 = tf.reduce_mean(tf.square(diff))
    
    V = model(y_boundary, x_boundary)
    diff = V - u_boundary(x_boundary, y_boundary)
    L2 = tf.reduce_mean(tf.square(diff))
                
    return L1, L2

def loss_DRM(model, x_interior, y_interior, x_boundary, y_boundary):
    V = model(y_interior,x_interior)
    V_y = tf.gradients(V, y_interior)[0]
    V_x = tf.gradients(V, x_interior)[0]
    
    diff = (V_x**2 + V_y**2)/2 - f(x_interior,y_interior) *V 
    L1 = tf.reduce_mean(diff)
    
    V = model(y_boundary, x_boundary)
    diff = V - u_boundary(x_boundary, y_boundary)
    L2 = tf.reduce_mean(tf.square(diff))
                
    return L1, L2

def loss_DDM(model, x_interior, y_interior, x_boundary, y_boundary):
    V = model(y_interior,x_interior)
    V_yy = tf.gradients(V[:,1],y_interior)[0]
    V_xx = tf.gradients(V[:,2],x_interior)[0]
    diff = V_yy + V_xx + f(x_interior,y_interior)
    L1 = tf.reduce_mean(tf.square(diff))
    cons1 = tf.gradients(V[:,0], y_interior)[0] - tf.reshape(V[:,1],[-1,1])
    cons2 = tf.gradients(V[:,0], x_interior)[0] - tf.reshape(V[:,2],[-1,1])    
    L1 = L1 + tf.reduce_mean(tf.square(cons1)) + tf.reduce_mean(tf.square(cons2)) 
    
    V = model(y_boundary, x_boundary)
    diff = tf.reshape(V[:,0],[-1,1]) - u_boundary(x_boundary, y_boundary)
    L2 = tf.reduce_mean(tf.square(diff))
    return L1, L2
#%% Set up network

# initialize Model
model_DGM = net.DeepNet(nodes_per_layer, num_layers, dim,final_trans="tanh")
model_DRM = net.DeepNet(nodes_per_layer, num_layers, dim,final_trans="tanh")
model_DDM = net.DeepNet_DDM(nodes_per_layer, num_layers, dim,final_trans="tanh", k = 2)

# tensor placeholder (_tnsr suffix indicates tensors)
x_e_tnsr = tf.placeholder(tf.float32, [None, 1])
y_e_tnsr = tf.placeholder(tf.float32, [None, 1])
x_b_tnsr = tf.placeholder(tf.float32, [None, 1])
y_b_tnsr = tf.placeholder(tf.float32, [None, 1])

# loss
L1_DDM, L2_DDM = loss_DDM(model_DDM, x_e_tnsr, y_e_tnsr, x_b_tnsr, y_b_tnsr)
loss_DDM = L1_DDM + L2_DDM
#
L1_DGM, L2_DGM = loss_DGM(model_DGM,x_e_tnsr, y_e_tnsr, x_b_tnsr, y_b_tnsr)
loss_DGM = L1_DGM + L2_DGM


L1_DRM, L2_DRM = loss_DRM(model_DRM,x_e_tnsr, y_e_tnsr, x_b_tnsr, y_b_tnsr)
loss_DRM = L1_DRM + L2_DRM

# Solution
#V = model(y_e_tnsr, x_e_tnsr)

# optimizer
optimizer_DDM = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_DDM)

optimizer_DGM = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_DGM)

optimizer_DRM = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_DRM)



# initialize variables
init_op = tf.global_variables_initializer()

# open session
sess = tf.Session()
sess.run(init_op)


#%% Train network

# record the error in every plot
error_record = []

step = 0
for i in range(sampling_stages):
    # sample first
    y_e, x_e, y_b, x_b = sampler(n_interior, n_boundary)

    # for each sample, take required steps with optimizer
    for _ in range(steps_per_sample):
        loss_r, L1, L2, _ = sess.run([loss_DDM, L1_DDM, L2_DDM, optimizer_DDM],
                                    feed_dict = {y_e_tnsr:y_e, 
                                                x_e_tnsr:x_e, 
                                                y_b_tnsr:y_b,
                                                x_b_tnsr:x_b,})
    
        step = step + 1
        if step%500 == 0:
            print(loss_r, L1, L2, step)
        loss_r, L1, L2, _ = sess.run([loss_DGM, L1_DGM, L2_DGM, optimizer_DGM],
                                    feed_dict = {y_e_tnsr:y_e, 
                                                x_e_tnsr:x_e, 
                                                y_b_tnsr:y_b,
                                                x_b_tnsr:x_b,})
        if step%500 == 0:
            print(loss_r, L1, L2, step)
        loss_r, L1, L2, _ = sess.run([loss_DRM, L1_DRM, L2_DRM, optimizer_DRM],
                                    feed_dict = {y_e_tnsr:y_e, 
                                                x_e_tnsr:x_e, 
                                                y_b_tnsr:y_b,
                                                x_b_tnsr:x_b,})
        
        if step%500 == 0:
            print(loss_r, L1, L2, step)
        
#%% plot the training process 
plt.figure()
plt.figure(figsize = (18,13))
N = 101
x_plot = np.linspace(a,b,N)
y_plot = np.linspace(a,b,N)
MAP = plt.cm.Spectral
norm = matplotlib.colors.Normalize(vmin=-1, vmax=1)
# time nodes for plot
XX,TT = np.meshgrid(x_plot,y_plot)

V_DGM = model_DGM(y_e_tnsr, x_e_tnsr)
V_DRM = model_DRM(y_e_tnsr, x_e_tnsr)
V_DDM = model_DDM(y_e_tnsr, x_e_tnsr)
test_input = [[[XX[i][j],TT[i][j]] for i in range(N)] for j in range(N)]
nodes = []
for i in test_input:
    nodes += i
nodes = np.array(nodes)
DGM_sol = sess.run([V_DGM], feed_dict = {y_e_tnsr:nodes[:,1].reshape(-1,1), x_e_tnsr:nodes[:,0].reshape(-1,1)})[0]
DRM_sol = sess.run([V_DRM], feed_dict = {y_e_tnsr:nodes[:,1].reshape(-1,1), x_e_tnsr:nodes[:,0].reshape(-1,1)})[0]
DDM_sol = sess.run([V_DDM], feed_dict = {y_e_tnsr:nodes[:,1].reshape(-1,1), x_e_tnsr:nodes[:,0].reshape(-1,1)})[0][:,0]

plt.subplot(3,3,1)
plt.contourf(XX,TT,DGM_sol.reshape(N,N).T,cmap=MAP, norm=norm)
plt.title('DGM solution')
plt.colorbar()
plt.subplot(3,3,2)

#tmp = (np.abs(predict_sol)<=1)*predict_sol + np.sign(predict_sol)*(np.abs(predict_sol)>1)
fig2 = plt.contourf(XX,TT,DRM_sol.reshape(N,N).T,cmap=MAP,norm=norm)
plt.title('DRM solution')
plt.colorbar()

plt.subplot(3,3,3)
plt.contourf(XX,TT,DDM_sol.reshape(N,N).T,cmap=MAP, norm=norm)
plt.title('DDM solution')
plt.colorbar()
plt.subplot(3,3,4)
plt.contourf(XX,TT,np.abs(DGM_sol.reshape(N,N).T - ue(XX,TT)),cmap=MAP)
plt.title('DGM error')
plt.colorbar()
plt.subplot(3,3,5)
plt.contourf(XX,TT,np.abs(DRM_sol.reshape(N,N).T-ue(XX,TT)),cmap=MAP)
plt.title('DRM error')
plt.colorbar()
plt.subplot(3,3,6)
plt.contourf(XX,TT,np.abs(DDM_sol.reshape(N,N).T-ue(XX,TT)),cmap=MAP)
plt.title('DDM error')
plt.colorbar()

