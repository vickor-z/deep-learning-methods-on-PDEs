# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 13:22:48 2020

@author: vickor

SUSTech
"""
# DGM based solver for heat equation

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
import loss as ls

#%% Parameters 
pi = 4 * np.arctan(1.0)

# Solution domain
t, T = 0.0, 1.0
dt = 0.001
a, b = 0.0, 2*pi
dim = 1

# Neural network parameters
num_layers = 3
nodes_per_layer = 50
learning_rate = 0.001

# Training and Sampling parameters
sampling_stages = 500
steps_per_sample = 5
n_interior = 200
n_initial = 50
n_boundary = 50


#%% True solution, initial condition and boundary condition
def ue(x, t):
    ans = np.exp(-t) * np.sin(x)
    return ans

def u_initial(x):
    ans = tf.sin(x)
    return ans

def u_boundary(x,t):
    return tf.constant(0.0)

#%% Set up network

# initialize Model
model = net.DeepNet(nodes_per_layer, num_layers, dim)


# tensor placeholder (_tnsr suffix indicates tensors)
t_e_tnsr = tf.placeholder(tf.float32, [None, 1])
x_e_tnsr = tf.placeholder(tf.float32, [None, 1])
t_i_tnsr = tf.placeholder(tf.float32, [None, 1])
x_i_tnsr = tf.placeholder(tf.float32, [None, 1])
t_b_tnsr = tf.placeholder(tf.float32, [None, 1])
x_b_tnsr = tf.placeholder(tf.float32, [None, 1])

# loss
L1_tnsr, L2_tnsr, L3_tnsr = ls.loss(model, t_e_tnsr, x_e_tnsr, t_i_tnsr, x_i_tnsr, t_b_tnsr, x_b_tnsr, u_initial, u_boundary).Heat()
loss_tnsr = L1_tnsr + L2_tnsr + L3_tnsr

# Solution
V = model(t_e_tnsr, x_e_tnsr)

# optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_tnsr)

# initialize variables
init_op = tf.global_variables_initializer()

# open session
sess = tf.Session()
sess.run(init_op)


#%% Train network

step = 0
for i in range(sampling_stages):
    # sample first
    t_e, x_e, t_i, x_i, t_b, x_b = ls.sampler(n_interior, n_initial, n_boundary, a, b, t, T, dim)

    # for each sample, take required steps with optimizer
    for _ in range(steps_per_sample):
        loss, L1, L2, L3, _ = sess.run([loss_tnsr, L1_tnsr, L2_tnsr, L3_tnsr, optimizer],
                                    feed_dict = {t_e_tnsr:t_e, 
                                                x_e_tnsr:x_e, 
                                                t_i_tnsr:t_i, 
                                                x_i_tnsr:x_i,
                                                t_b_tnsr:t_b,
                                                x_b_tnsr:x_b,})
    
        step = step + 1
        if step%100 == 0:
            print(loss, L1, L2, L3, step)

#%% numerical method
N = 101
x_plot = np.linspace(a,b,N)
t = 0
T = 1
dt = 0.01
t_plot = np.linspace(t,T,N)
h = (b-a)/(N-1)
A = np.eye(N,k=1)-2*np.eye(N)+np.eye(N,k=-1)
B = np.eye(N) - dt/h**2 * A
B[0,:],B[0,0]=0,1
B[N-1,:],B[N-1,N-1]=0,1 
u0 = np.sin(x_plot)
u = u0
u_record = []
u_record.append(u0)
while t<T:
    u = np.linalg.solve(B,u)
    u_record.append(u)
    t = t + dt


#%% plot the training process 
plt.figure()
plt.figure(figsize = (18,7))
MAP = plt.cm.Spectral
norm = matplotlib.colors.Normalize(vmin=-1, vmax=1)
# time nodes for plot
XX,TT = np.meshgrid(x_plot,t_plot)


test_input = [[[XX[i][j],TT[i][j]] for i in range(N)] for j in range(N)]
nodes = []
for i in test_input:
    nodes += i
nodes = np.array(nodes)
predict_sol = sess.run([V], feed_dict = {t_e_tnsr:nodes[:,1].reshape(-1,1), x_e_tnsr:nodes[:,0].reshape(-1,1)})


plt.subplot(2,3,1)
plt.contourf(XX,TT,np.array(u_record),cmap=MAP, norm=norm)
plt.title('FD solution')
plt.colorbar()
plt.subplot(2,3,2)
predict_sol = predict_sol[0][:,0]
tmp = (np.abs(predict_sol)<=1)*predict_sol + np.sign(predict_sol)*(np.abs(predict_sol)>1)
fig2 = plt.contourf(XX,TT,tmp.reshape(N,N).T,cmap=MAP,norm=norm)
plt.title('DL solution')
plt.colorbar()

plt.subplot(2,3,3)
plt.contourf(XX,TT,ue(XX,TT),cmap=MAP, norm=norm)
plt.title('exact solution')
plt.colorbar()
plt.subplot(2,3,4)
plt.contourf(XX,TT,np.abs(np.array(u_record) - ue(XX,TT)),cmap=MAP)
plt.title('FD error')
plt.colorbar()
plt.subplot(2,3,5)
plt.contourf(XX,TT,np.abs(predict_sol.reshape(N,N).T-ue(XX,TT)),cmap=MAP)
plt.title('DL error')
plt.colorbar()