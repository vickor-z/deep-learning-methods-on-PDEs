# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 18:45:31 2019

@author: vickor
"""

# DDM based solver for kdv equation

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
import time

#%% Parameters for Allen-Cahn
eps = 0.005
alpha = 0.5
pi = 4 * np.arctan(1.0)
mult = 1
nu = 0.05

# Solution domain
t, T = 0.0, 1.0
dt = 0.001
a, b = -2.0, 2.0
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
    ans = np.tanh(x+2*t-1)
    return ans

def u_initial(x):
    ans = tf.tanh(x -1)
    return ans

def u_boundary(x,t):
    return tf.tanh(x+2*t-1)

#%% Set up network
model = net.DeepNet_DDM(nodes_per_layer, num_layers, dim, k = 2,final_trans="tanh")

# tensor placeholder (_tnsr suffix indicates tensors)
t_e_tnsr = tf.placeholder(tf.float32, [None, 1])
x_e_tnsr = tf.placeholder(tf.float32, [None, 1])
t_i_tnsr = tf.placeholder(tf.float32, [None, 1])
x_i_tnsr = tf.placeholder(tf.float32, [None, 1])
t_b_tnsr = tf.placeholder(tf.float32, [None, 1])
x_b_tnsr = tf.placeholder(tf.float32, [None, 1])

# loss
L1_DDM, L2_DDM, L3_DDM = ls.loss(model, t_e_tnsr, x_e_tnsr, t_i_tnsr, x_i_tnsr, t_b_tnsr, x_b_tnsr, u_initial, u_boundary).kdv_DDM()
loss_DDM = L1_DDM + L2_DDM + L3_DDM

# Solution
V = model(t_e_tnsr, x_e_tnsr)

# optimizer
optimizer_DDM = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_DDM)

#np.random.seed(seed=0)

# initialize variables
init_op = tf.global_variables_initializer()

# open session
sess = tf.Session()
sess.run(init_op)
#%% preparation
N = 101
x_plot = np.linspace(a,b,N)
t = 0
T = 1
t_plot = np.linspace(t,T,N)
XX,TT = np.meshgrid(x_plot,t_plot)
exact_sol = ue(XX,TT)

test_input = [[[XX[i][j],TT[i][j]] for i in range(N)] for j in range(N)]
nodes = []
for i in test_input:
    nodes += i
nodes = np.array(nodes)

error_record = []
loss_record = []
time_record = []
step = 0
time1 = 0
#%% Train network


# record the error in every plot
error_record = []
loss_record = []

step = 0
for i in range(sampling_stages):
    # sample first
    t_e, x_e, t_i, x_i, t_b, x_b = ls.sampler(n_interior, n_initial, n_boundary, a, b, t, T, dim)

    # for each sample, take required steps with optimizer
    for _ in range(steps_per_sample):
        t1 = time.time()
        loss, L1, L2, L3, _ = sess.run([loss_DDM, L1_DDM, L2_DDM, L3_DDM, optimizer_DDM],
                                    feed_dict = {t_e_tnsr:t_e, 
                                                x_e_tnsr:x_e, 
                                                t_i_tnsr:t_i, 
                                                x_i_tnsr:x_i,
                                                t_b_tnsr:t_b,
                                                x_b_tnsr:x_b,})
        t2 = time.time()
        time1 = time1 + t2-t1
        step = step + 1
        if step%100 == 0:
            print(loss, L1, L2, L3, step)
            loss_record.append(loss)
            DDM_sol = sess.run([V], feed_dict = {t_e_tnsr:nodes[:,1].reshape(-1,1), x_e_tnsr:nodes[:,0].reshape(-1,1)})[0][:,0]
            exact_sol = ue(XX,TT)
            exact_sol = np.reshape(exact_sol.T,[-1,])
            er = DDM_sol - exact_sol
            error_record.append(np.linalg.norm(er)/np.linalg.norm(exact_sol))
            time_record.append(time1)
#%% numerical method



#%% plot the training process 
plt.figure()
plt.figure(figsize = (12,8))
MAP = plt.cm.Spectral
norm = matplotlib.colors.Normalize(vmin=-1, vmax=1)
# time nodes for plot
XX,TT = np.meshgrid(x_plot,t_plot)


test_input = [[[XX[i][j],TT[i][j]] for i in range(N)] for j in range(N)]
nodes = []
for i in test_input:
    nodes += i
nodes = np.array(nodes)
DDM_sol = sess.run([V], feed_dict = {t_e_tnsr:nodes[:,1].reshape(-1,1), x_e_tnsr:nodes[:,0].reshape(-1,1)})[0][:,0]

ftsize = 'xx-large'
plt.subplot(2,2,1)
plt.tick_params(labelsize='large')
fig2 = plt.contourf(XX,TT,DDM_sol.reshape(N,N).T,cmap=MAP,norm=norm)
plt.title('LDGM solution',fontsize=ftsize)
plt.colorbar()

plt.subplot(2,2,2)
plt.tick_params(labelsize='large')
plt.contourf(XX,TT,ue(XX,TT),cmap=MAP, norm=norm)
plt.title('exact solution',fontsize=ftsize)
plt.colorbar()
plt.subplot(2,2,3)
plt.tick_params(labelsize='large')
plt.contourf(XX,TT,np.abs(DDM_sol.reshape(N,N).T-ue(XX,TT)),cmap=MAP)
plt.title('LDGM error',fontsize=ftsize)
plt.colorbar()
plt.subplot(2,2,4)
plt.tick_params(labelsize='large')
total_it = sampling_stages*steps_per_sample
it = np.linspace(0,total_it,total_it//100)
plt.plot(it, time_record,'r',label='DDM')
plt.legend(fontsize='x-large')
plt.xlabel('iteration steps',fontsize='x-large')
plt.ylabel('CPUtime/s',fontsize='x-large')
#plt.savefig("kdv_equation_contourf.eps",format="eps")
#f = open('neural.csv','a')
#writer = csv.writer(f)
#data = [error_record,loss_record]
#writer.writerow(data)
#f.close()