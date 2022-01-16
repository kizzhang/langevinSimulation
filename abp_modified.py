# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 12:08:47 2021

@author: Kaneki
"""
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 17:28:36 2021

@author: Kaneki
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def Periodicity(pos,l):
    if pos >= -l and pos <= l :
        return pos
    elif pos < -l:
        return pos + 2*l
    elif pos > l:
        return pos - 2*l

def ABP_move(t, dt, N, crit_coll_num, l):
    coll_num = np.zeros((N, int(t/dt)))
    for i in range(0, int(t/dt) - 1): # time evolution
        # Collision
        for p1 in range(0,N):
            for p2 in range(p1,N):
                if p1 == p2:
                    continue
                
                # Collision criteria
                r = np.sqrt((x[p1,i] - x[p2,i]) ** 2 + (y[p1,i] - y[p2,i]) ** 2)
                if r > 2.1 * a:
                    continue
                else:
                    coll_num[p1,i] += 1
                    coll_num[p2,i] += 1
                    
                    
        for dum in range(len(coll_num)):
            if coll_num[dum, i] >= crit_coll_num:
                theta[dum,i] = theta[dum,i] + np.random.uniform(0,2*np.pi) # a random angle to avoid coll 
                dx = v * np.cos(theta[dum,i]) * dt + np.sqrt(2*Dt*dt) * np.random.randn()
                dy = v * np.sin(theta[dum,i]) * dt + np.sqrt(2*Dt*dt) * np.random.randn()
                
                
                x_new = x[dum,i] + dx
                y_new = y[dum,i] + dy
                
                theta[dum,i+1] = theta[dum,i] + np.sqrt(2*Dr*dt) * np.random.randn()

                # Periodic boundary condition on x  
                x[dum,i+1] = Periodicity(x_new, l)
            
                # Periodic boundary condition on y     
                y[dum,i+1] = Periodicity(y_new, l)
                
                # x position if there is no jump
                x_nojump[dum,i+1] = x_nojump[dum,i] + dx 
                
                # y position if there is no jump
                y_nojump[dum,i+1] = y_nojump[dum,i] + dy
            
            else:
                dx = v * np.cos(theta[dum,i]) * dt + np.sqrt(2*Dt*dt) * np.random.randn()
                dy = v * np.sin(theta[dum,i]) * dt + np.sqrt(2*Dt*dt) * np.random.randn()
                
                x_new = x[dum,i] + dx
                y_new = y[dum,i] + dy
                
                theta[dum,i+1] = theta[dum,i] + np.sqrt(2*Dr*dt) * np.random.randn() 
                
                # Periodic boundary condition on x  
                x[dum,i+1] = Periodicity(x_new, l)
            
                # Periodic boundary condition on x      
                y[dum,i+1] = Periodicity(y_new,l)
                
                # x position if there is no jump
                x_nojump[dum,i+1] = x_nojump[dum,i] + dx 
                
                # y position if there is no jump
                y_nojump[dum,i+1] = y_nojump[dum,i] + dy
            
        print("Time Step: ", i)
    return x, y, theta, coll_num

# CONSTANTS

v = 3.12e-5                # swimming speed of B. Subtilis [m/s]
k = 1.38e-23               # Boltzmann constant [m^2kg/s^2K]
T = 293                    # Room temperature [K]
eta = 1e-3                  # viscosity of water [Pa s]
a = 2e-6                   # spherical cell radius [m]
Dr = k*T/(8*np.pi*eta*a**3)   # rotational diffusion coefficient of B. Subtilis
Dt = k*T/(6*np.pi*eta*a)   # translation diffusion coefficient of B. Subtilis

# ADJUSTABLE PARAMETERS

t = 10    # time over which motion is observed [s]
dt = 0.01      # time step between recorded positions
N = 900  # number of cells 
crit_coll_num = 1 # number of collisions a bacetrium will walk away
l = 0.5 * 1e-4 # box width
psi = N * np.pi * a**2 / (2*l)**2 # packing fraction


# INITIAL CONDITIONS

theta = np.zeros((N,int(t/dt)))  # initial swimming orientation [radians]
x = np.zeros((N,int(t/dt)))  # initial x position [m]
y = np.zeros((N,int(t/dt)))    # initial y position [m]
x_nojump = np.zeros((N,int(t/dt))) # x position without jump
y_nojump = np.zeros((N,int(t/dt))) # y position without jump

# Initializing x y theta
for n in range(N):  
    # x positions
    x[n,0] = np.random.uniform(-l,l)
    x_nojump[n,0] = x[n,0]
    
    # y positions
    y[n,0] = np.random.uniform(-l,l)
    y_nojump[n,0] = y[n,0]
    
    theta[n,0] = np.random.uniform(-2*np.pi, 2*np.pi)
    
    
x,y,_,col_num = ABP_move(t,dt,N,crit_coll_num,l)
print("Packing Fraction = ", psi)

'''
import pandas as pd
df_x = pd.DataFrame(x)
df_y = pd.DataFrame(y)
df_x_non_p = pd.DataFrame(x_nojump)
df_y_non_p = pd.DataFrame(y_nojump)


df_x.to_csv('x_p.dat')
df_y.to_csv('y_p.dat')
df_x_non_p.to_csv('x_nonp.dat')
df_y_non_p.to_csv('y_nonp.dat')
'''

# MAIN SCRIPT

fig = plt.figure(dpi = 141)
ax = plt.axes(xlim=(-1*l, 1*l), ylim=(-1*l, 1*l))
ax.set_aspect(1)
fig.canvas.draw()

s = (ax.get_window_extent().width * 72./fig.dpi * a / l)**2

scat = ax.scatter([], [], s)
scat1 = ax.scatter([], [], s)   

def animation(frame):
    data = np.hstack((x[:,frame, np.newaxis], y[:,frame, np.newaxis]))
    scat.set_offsets(data)
    return scat,
    
def animation_non_Periodic(frame):
    data1 = np.hstack((x_nojump[:,frame, np.newaxis], y_nojump[:,frame, np.newaxis]))
    scat1.set_offsets(data1)
    return scat1,

def animation_with_trajectory(frame):
    ax.cla()
    for i in range(N):
        ax.plot(x[i,:frame], y[i,:frame], linestyle = '-', color = 'blue')
        ax.plot(x[i,frame], y[i,frame], 'ro')
    
    ax.set_xlim(-l,l)
    ax.set_ylim(-l,l)

ani = FuncAnimation(fig, animation, frames=range(int(t/dt)),\
                    interval = 10, repeat=False)

ani.save("movie2.mp4", fps = 40)
