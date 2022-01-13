# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 18:45:31 2019

@author: vickor
"""

# DDM based solver for high dimensional heat equation

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
a, b = 0.0, 1.0
dim = 5

# Neural network parameters
num_layers = 3
nodes_per_layer = 50
learning_rate = 0.001

# Training and Sampling parameters
sampling_stages = 10000
steps_per_sample = 5
n_interior = 200
n_initial = 50
n_boundary = 50 

c = 1
l1 = 1
l2 = 1
#%% True solution, initial condition and boundary condition
def ue(x,t):
    y = c*np.sum(x*(1-x),axis=1) * (t+1)
#    y = 0.4*x*(1-x)*(t+1)
    return y


def u_boundary(x,t):
    y = c*tf.reduce_sum(x*(1-x),axis=1) 
    y = tf.reshape(y,[-1,1]) * (t+1)
    return y

def u_initial(x):
    ans = c*tf.reduce_sum(x*(1-x),axis=1)
    ans = tf.reshape(ans,[-1,1])
    return ans

def f(x,t):
    r = tf.reduce_sum(x*(1-x),axis=1)
    ans = tf.reshape(r,[-1,1]) + 2*dim*(t+1)
    ans = c*tf.reshape(ans,[-1,1])
    return ans

#%% Set up network
def linear(features, name=None):
    return features

# initialize Model
final_s = linear
model = net.DeepNet_2VND(nodes_per_layer, num_layers, dim,final_trans=final_s)
t_e_tnsr = tf.placeholder(tf.float32, [None, 1])
x_e_tnsr = tf.placeholder(tf.float32, [None, dim])
t_i_tnsr = tf.placeholder(tf.float32, [None, 1])
x_i_tnsr = tf.placeholder(tf.float32, [None, dim])
t_b_tnsr = tf.placeholder(tf.float32, [None, 1])
x_b_tnsr = tf.placeholder(tf.float32, [None, dim])

# loss
L1_tnsr, L2_tnsr, L3_tnsr = ls.loss(model, t_e_tnsr, x_e_tnsr, t_i_tnsr, x_i_tnsr, t_b_tnsr, x_b_tnsr, u_initial, u_boundary).Heat_DDM_ND(f, dim, l1, l2)
loss_tnsr = L1_tnsr + L2_tnsr + L3_tnsr

# Solution
V = model(t_e_tnsr, x_e_tnsr)

# optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_tnsr)

# initialize variables
np.random.seed(seed = 0)
init_op = tf.global_variables_initializer()

# open session
sess = tf.Session()
sess.run(init_op)

#%%#%% preparation
N = 100
t = 0.0
T = 1.0

x = np.linspace(a,b,N+1)
t = np.linspace(t,T,N+1)
XX, TT = np.meshgrid(x,t)
#plt.subplot(2,3,1)
x = np.reshape(XX,[(N+1)**2,1])
y = np.zeros([(N+1)**2,dim-1])
x = np.hstack([x,y])
exact_sol = ue(x,np.reshape(TT,[(N+1)**2,]))
t = 0.0
T = 1.0
#%% Train network

# record the error in every plot
error_record = []
loss_record = []
time_record = []
data_record = []
step = 0
time1 = 0
for i in range(sampling_stages):
    # sample first
    t_e, x_e, t_i, x_i, t_b, x_b = ls.sampler(n_interior, n_initial, n_boundary, a, b, t, T, dim)

    # for each sample, take required steps with optimizer
    for _ in range(steps_per_sample):
        t1 = time.time()
        loss, L1, L2, L3, _ = sess.run([loss_tnsr, L1_tnsr, L2_tnsr, L3_tnsr, optimizer],
                                    feed_dict = {t_e_tnsr:t_e, 
                                                x_e_tnsr:x_e, 
                                                t_i_tnsr:t_i, 
                                                x_i_tnsr:x_i,
                                                t_b_tnsr:t_b,
                                                x_b_tnsr:x_b,})
        t2 = time.time()    
        time1 = time1 + t2 - t1
        step = step + 1
        if step%1000 == 0:
            print(loss, L1, L2, L3, step)
            loss_record.append(loss)
#            exact_sol = ue(x_e,t_e)
            predict_sol = sess.run([V], feed_dict = {t_e_tnsr:np.reshape(TT,[(N+1)**2,1]), x_e_tnsr:x})[0][:,0]
            er = predict_sol - exact_sol
            error_record.append(np.linalg.norm(er)/np.linalg.norm(exact_sol))
            data_record.append(predict_sol)
            time_record.append(time1)
#    if loss < 1e-3:
#        print(loss, L1, L2, L3, step)
#        break
#%% numerical method
#N = 101
#x_plot = np.linspace(a,b,N)
#t = 0
#T = 1
#dt = 0.01
#t_plot = np.linspace(t,T,N)
#h = (b-a)/(N-1)
#A = np.eye(N,k=1)-2*np.eye(N)+np.eye(N,k=-1)
#B = np.eye(N) - dt/h**2 * A
#B[0,:],B[0,0]=0,1
#B[N-1,:],B[N-1,N-1]=0,1 
#u0 = np.sin(x_plot)
#u = u0
#u_record = []
#u_record.append(u0)
#while t<T:
#    u = np.linalg.solve(B,u)
#    u_record.append(u)
#    t = t + dt


#%% numerical method
MAP = plt.cm.Spectral
norm = matplotlib.colors.Normalize(vmin=0, vmax=0.6)
plt.figure(figsize=(12,3.5))
levels = np.arange(0,0.9,0.1)
N = 100
t = 0.0
T = 1.0

x = np.linspace(a,b,N+1)
t = np.linspace(0,T,N+1)
XX, TT = np.meshgrid(x,t)
plt.subplot(1,2,1)
x = np.reshape(XX,[(N+1)**2,1])
y = np.zeros([(N+1)**2,dim-1])
x = np.hstack([x,y])
exact_sol = ue(x,np.reshape(TT,[(N+1)**2,]))
plt.contourf(XX,TT,np.reshape(exact_sol,[N+1,N+1]),cmap=MAP,norm=norm, levels = levels)
plt.xlabel(r"$x_1$")
plt.ylabel("t")

plt.title('exact solution')
plt.colorbar()

ax2 = plt.subplot(1,2,2)
x = np.reshape(XX,[(N+1)**2,1])
y = np.zeros([(N+1)**2,dim-1])
x = np.hstack([x,y])
predict_sol = sess.run([V],feed_dict = {t_e_tnsr:np.reshape(TT,[(N+1)**2,1]), x_e_tnsr:x})[0][:,0]
#predict_sol = predict_sol * (predict_sol>0)
plt.contourf(XX,TT,np.reshape(predict_sol,[N+1,N+1]),cmap=MAP,norm=norm, levels = levels)
plt.title('LDGM solution')
plt.xlabel(r"$x_1$")
plt.ylabel("t")
plt.colorbar()



#plt.savefig("d=%s.eps"%(dim),format="eps")

#%%
k = [1,2,10,20,50]
l = len(k)

plt.figure(figsize=(12,3.5*l))
i = 0
ftsize = 20
plt.subplot(l,2,2*i+1)
plt.tick_params(labelsize='large')
plt.contourf(XX,TT,np.reshape(exact_sol,[N+1,N+1]), cmap=MAP, norm=norm, levels=levels)
plt.colorbar()   
plt.title('exact solution', fontsize=ftsize)
plt.ylabel('k=%d'%(1000), fontsize=ftsize)
plt.subplot(l,2,2*i+2)
plt.tick_params(labelsize='large')
plt.contourf(XX,TT,np.reshape(data_record[k[i]-1],[N+1,N+1]), cmap=MAP, norm=norm, levels=levels)
plt.colorbar()
plt.title('LDGM solution', fontsize=ftsize)


for i in range(1, l):
    plt.subplot(l,2,2*i+1)
    plt.tick_params(labelsize='large')
    plt.contourf(XX,TT,np.reshape(exact_sol,[N+1,N+1]), cmap=MAP, norm=norm, levels=levels)
    plt.colorbar()
    plt.ylabel('k=%d'%(1000*k[i]), fontsize=ftsize)
    plt.subplot(l,2,2*i+2)
    plt.tick_params(labelsize='large')
    plt.contourf(XX,TT,np.reshape(data_record[k[i]-1],[N+1,N+1]), cmap=MAP, norm=norm, levels=levels)
    plt.colorbar()

#plt.savefig('nd_heat.eps',format='eps')
#%%
plt.figure(figsize=(19,5))
it = np.linspace(1000, sampling_stages*steps_per_sample, int(sampling_stages*steps_per_sample/1000))
ftsize = 'xx-large'
plt.rc('legend',fontsize=ftsize)
plt.subplot(1,3,1)
plt.tick_params(labelsize='large')
plt.semilogy(it,loss_record,'r',label='LDGM')
#plt.legend()
plt.xlabel('iteration steps',fontsize=ftsize)
plt.ylabel('loss',fontsize=ftsize)
plt.subplot(1,3,2)
plt.tick_params(labelsize='large')
plt.plot(it, time_record,'r',label='LDGM')
#plt.legend()
plt.xlabel('iteration steps',fontsize=ftsize)
plt.ylabel('CPU time/s',fontsize=ftsize)
plt.subplot(1,3,3)
plt.tick_params(labelsize='large')
plt.semilogy(time_record,error_record,'r',label='LDGM')
plt.legend()
plt.xlabel('CPU time/s',fontsize=ftsize)
plt.ylabel('relative L2 error',fontsize=ftsize)
#plt.savefig('heat_loss_time.eps',format='eps')