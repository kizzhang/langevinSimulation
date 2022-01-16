# -*- coding: utf-8 -*-
"""
Created on Sat Dec  4 20:54:43 2021

@author: Kaneki
"""
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 13:25:10 2021

@author: Kaneki
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.spatial import distance
from numpy.linalg import norm, det


def Periodicity(pos,l):
    if pos >= -l and pos <= l :
        return pos
    elif pos < -l:
        return pos + 2*l
    elif pos > l:
        return pos - 2*l
    
def Populate_distance_matrix(N,i):
    Pos_vec = []
    for num in range(N):     
        Pos_vec.append((x[num,i],y[num,i]))
    Dist_mat = distance.cdist(Pos_vec,Pos_vec, metric = 'euclidean')
         
    return Dist_mat

def Angle(vec): return np.arctan2(vec[1],np.dot(vec,[1,0]))

def Angle_btw_2_vecs(vec1, vec2): return np.arctan2(det([vec1,vec2]),np.dot(vec1,vec2))

def Boundary_condition(b_type,x,y, x_nojump,y_nojump,dx, dy, par,i):
    if b_type == "periodic":
        x_new = x[par,i] + dx
        y_new = y[par,i] + dy
        # Periodic boundary condition on x  
        x[par,i+1] = Periodicity(x_new, l)
            
        # Periodic boundary condition on y     
        y[par,i+1] = Periodicity(y_new, l)
                
        # x position if there is no jump
        x_nojump[par,i+1] = x_nojump[par,i] + dx 
                
        # y position if there is no jump
        y_nojump[par,i+1] = y_nojump[par,i] + dy
        
    else:
        print("Wrong input")
        x=1
        return x

def ABP_move(t, dt, N, l): 
    for i in range(0, int(t/dt)-1): # time evolution
        Dist_mat = Populate_distance_matrix(N, i)  # Distance mattrice for one instance of time
        #print(Dist_mat)
        for p1 in range(N):
            # Increment initializing for theta,x,y
            radial_vec = [x[p1,i],y[p1,i]] / np.sqrt((x[p1,i]**2+y[p1,i]**2))
            theta_p1_new = np.sqrt(2*Dr*dt) * np.random.randn() +\
                        G_attr * np.sin(Angle_btw_2_vecs([np.cos(theta[p1,i]), np.sin(theta[p1,i])],-1*radial_vec))\
                           *dt * 1 / R_vor
            
            dx_p1_new = v * np.cos(theta[p1,i]) * dt#  - G_attr * x[p1,i] / np.sqrt((x[p1,i]**2+y[p1,i]**2)) *dt #+ np.sqrt(2*Dt*dt) * np.random.randn()
            dy_p1_new = v * np.sin(theta[p1,i]) * dt#  - G_attr * y[p1,i] / np.sqrt((x[p1,i]**2+y[p1,i]**2)) *dt#+ np.sqrt(2*Dt*dt) * np.random.randn()
           
            cluster_num = 0
            for p2 in range(N):
                if Dist_mat[p1,p2] < epsilon_int:
                    cluster_num +=1  
                    
            for p2 in range(N):
                if p1 == p2:
                    continue
                
                r = Dist_mat[p1,p2] # r=rji = rij is distance between pairs
                #### Interaction Region ####
                if r > epsilon_int:
                    continue
                r''' 
                # Positional exclusion
                if r < zeta_ex:
                    dx_p1_new = dx_p1_new - k_pos * (x[p2,i] - x[p1,i]) * dt
                    dy_p1_new = dy_p1_new - k_pos * (y[p2,i] - y[p1,i]) * dt 
                   
                # Angular exclusion
                if r <= epsilon_ex: # Exclusion region
                    alpha_ji = Angle([x[p2,i]-x[p1,i], y[p2,i]-y[p1,i]])
                    
                    theta_p1_new = theta_p1_new - k_rad * np.sin(alpha_ji-theta[p1,i]) * dt
              '''
                # Angular Alignment
                if r <= epsilon_aa: # Alignment region
                    alig_incre_p1 = mu_plus * dt * np.sin(theta[p2,i] - theta[p1,i]) # * (1-(r/epsilon_aa)**2) 
                    
                    theta_p1_new = theta_p1_new + alig_incre_p1/ cluster_num
                r'''      
                # Angular Disalignment
                if r > epsilon_aa: # Anti-alignment region
                    anti_incre_p1 =  mu_minus * 4 * (r-epsilon_aa)*(epsilon_int-r)/ \
                                        (epsilon_int - epsilon_aa)**2 * dt * np.sin(theta[p2,i] - theta[p1,i])
                                        
                    theta_p1_new = theta_p1_new - anti_incre_p1
                '''
            # Adding total increament of potential from all other particles to p1 (ANGULAR)
            x[p1, i+1] = x[p1, i+1] + x[p1, i] + dx_p1_new
            y[p1, i+1] = y[p1, i+1] + y[p1, i] + dy_p1_new
            theta[p1,i+1] =  theta[p1,i+1] + theta[p1,i] + theta_p1_new
            
            Boundary_condition('periodic', x,y, x_nojump,y_nojump,dx_p1_new, dy_p1_new, p1,i)
            
            
        if i ==1:
            print(Dist_mat)
        print("Time Step: ", i)
    return x, y, theta, vx, vy

# CONSTANTS
v = 1 #37                         # swimming speed of B. Subtilis [m/s]
Dr = 0.05 # 0.3 #   1                    # rotational diffusion coefficient of B. Subtilis

epsilon_int = 0.2 #30  # 1                  # beyound which interaction will be gone
epsilon_aa = 20 #    0.2             # allignment & antialignment transition radius
epsilon_ex = 6 #     0.1              # angular exclusion radius
zeta_ex =   3.5    # 0                 # positional exclusion radius

k_rad =  5   #      10                # anuglar exclusion strength
k_pos = 0.02  #    0                 # positional exclusion strength


# ADJUSTABLE PARAMETERS
t = 60      # time over which motion is observed [s]
dt = 0.01         # time step between recorded positions
N = 1000 #2000          # number of cells 
l =  4 # 5 #              # box width
G_attr = 10 # 0.05 #                       # gravitational pull 
R_vor = 10                          # gravitational radius 

mu_plus = 0.7  # 0.3  #4#  # alignment strength
mu_minus = 0.1  #0.004   # anti-alignment strength

# Packing fraction & density (Grossman et al. 2014 PRL)
psi = N * np.pi * epsilon_ex ** 2 / (2*l)**2 
rho = N / (2*l)**2

# INITIAL CONDITIONS
theta = np.zeros((N,int(t/dt)))  # initial swimming orientation [radians]
x = np.zeros((N,int(t/dt)))  # initial x position [m]
y = np.zeros((N,int(t/dt)))    # initial y position [m]
vx = np.zeros((N,int(t/dt)))
vy = np.zeros((N,int(t/dt)))
x_nojump = np.zeros((N,int(t/dt))) # x position without jump
y_nojump = np.zeros((N,int(t/dt))) # y position without jump

# Initializing x y theta; vx vy will be initialized in ABP move
for n in range(N):  
    # x positions
    x[n,0] = np.random.uniform(-l,l)
    x_nojump[n,0] = x[n,0]
    
    # y positions
    y[n,0] = np.random.uniform(-l,l)
    y_nojump[n,0] = y[n,0]
    
    theta[n,0] = np.random.uniform(-2*np.pi, 2*np.pi)
    
x,y,_,vx,vy = ABP_move(t,dt,N,l)

print("Packing Fraction = ", psi)
print("Density = ", rho)
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

fig = plt.figure(dpi = 180)
ax = plt.axes(xlim=(-1*l, 1*l), ylim=(-1*l, 1*l))
ax.set_aspect(1)
fig.canvas.draw()

s = (ax.get_window_extent().width * 72./fig.dpi * epsilon_ex / l)**2

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
    ax.clear()
    for i in range(N):
        ax.plot(x[i,:frame], y[i,:frame], linestyle = '-', color = 'blue')
        ax.plot(x[i,frame], y[i,frame], 'ro')
    
    ax.set_xlim(-l,l)
    ax.set_ylim(-l,l)
    
def animation_anisotropic(frame):
    ax.clear()
    for i in range(N):
       patch = plt.arrow(x[i,frame], y[i,frame], 0.04 * l *np.cos(theta[i,frame]),\
                             0.04 * l* np.sin(theta[i,frame]), width = 0,\
                                head_width = 0.35 * 0.05 *l,head_length = 0.4*0.05*l,\
                                color = 'k', overhang = 1, lw =  0.1)  
       ax.add_patch(patch)
       #ax.annotate("",xy = (x[i,frame], y[i,frame]),)
    ax.set_xlim(-l,l)
    ax.set_ylim(-l,l)

    
ani = FuncAnimation(fig, animation_anisotropic, frames= range(3000,5998),\
                    interval = 10, repeat=False)

ani.save("test1.mp4", fps = 100)
