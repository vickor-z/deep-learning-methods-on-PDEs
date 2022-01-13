# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 15:55:11 2021

@author: vickor
"""
#%%
import tensorflow as tf
import numpy as np
from tensorflow import keras as kr
from matplotlib import pyplot as plt

#%%
a, b = -1.0, 1.0
N = 101
x = np.linspace(a,b,N)
x = np.reshape(x,[-1,1])
c = np.pi
y = np.sin(c*x)

#%%
act = tf.nn.tanh
n = 32
model = kr.models.Sequential([kr.layers.Dense(n,activation=act,input_dim=1),
                              kr.layers.Dense(n,activation=act),
                              kr.layers.Dense(n,activation=act),
                              kr.layers.Dense(1)])

model.compile(optimizer = 'adam',
              loss = 'mse')
sess = tf.Session()
#%%
model.fit(x,y,epochs=2000)
#%%
y_p = model.predict(x)

#plt.plot(x,y,'r',x,y_p,'g')
#%%
grad = []
grad_f = []
grad.append(kr.backend.gradients(model.output,[model.input])[0])
grad_f.append(kr.backend.function(model.input,grad[0]))
for i in range(3):
    grad.append(kr.backend.gradients(grad[i],[model.input])[0])
    grad_f.append(kr.backend.function(model.input,grad[i+1]))
    


#%%
plt.figure(figsize=(13,9))
ftsize = 'x-large'
plt.rc('legend',fontsize=ftsize)
plt.subplot(2,2,1)
plt.tick_params(labelsize='large')
plt.plot(x,y,'r',label='u')
plt.plot(x,y_p,'g',label=r'$\varphi$')
plt.legend() 
plt.title('k=0',fontsize='xx-large')
plt.subplot(2,2,2)
plt.tick_params(labelsize='large')
grad_u = grad_f[0](x)
grad_y = np.pi*np.cos(np.pi*x)
#grad_y = c*(1-y**2)
plt.plot(x,grad_y,'r',label=r'$u_x$')
plt.plot(x,grad_u,'g',label=r'$\varphi_x$')
plt.legend()
plt.title('k=1',fontsize='xx-large')
plt.subplot(2,2,3)
plt.tick_params(labelsize='large')
grad2_u = grad_f[1](x)
grad2_y = -np.pi**2*np.sin(np.pi*x)
#grad2_y = c**2*(-2*y)*(1-y**2)
plt.plot(x,grad2_y,'r',label=r'$u_{xx}$')
plt.plot(x,grad2_u,'g',label=r'$\varphi_{xx}$')
plt.legend()
plt.title('k=2',fontsize='xx-large')
plt.subplot(2,2,4)
plt.tick_params(labelsize='large')
grad4_u = grad_f[3](x)
grad4_y = np.pi**4*np.sin(np.pi*x)
#grad4_y = c**4*8*y*(2-3*y**2)*(1-y**2)
plt.plot(x,grad4_y,'r',label=r'$u_{xxxx}$')
plt.plot(x,grad4_u,'g',label=r'$\varphi_{xxxx}$')
plt.legend()
plt.title('k=4',fontsize='xx-large')
plt.savefig('gradient_scale.eps',format='eps')