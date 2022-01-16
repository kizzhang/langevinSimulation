# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 09:53:44 2021

@author: Zhiyu
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from numpy.linalg import norm

def circle_line_segment_intersection(circle_center, circle_radius, pt1, pt2, full_line=True, tangent_tol=1e-13):
    """ Find the points at which a circle intersects a line-segment.  This can happen at 0, 1, or 2 points.

    :param circle_center: The (x, y) location of the circle center
    :param circle_radius: The radius of the circle
    :param pt1: The (x, y) location of the first point of the segment
    :param pt2: The (x, y) location of the second point of the segment
    :param full_line: True to find intersections along full line - not just in the segment.  False will just return intersections within the segment.
    :param tangent_tol: Numerical tolerance at which we decide the intersections are close enough to consider it a tangent
    :return Sequence[Tuple[float, float]]: A list of length 0, 1, or 2, where each element is a point at which the circle intercepts a line segment.

    Note: We follow: http://mathworld.wolfram.com/Circle-LineIntersection.html
    """

    (p1x, p1y), (p2x, p2y), (cx, cy) = pt1, pt2, circle_center
    (x1, y1), (x2, y2) = (p1x - cx, p1y - cy), (p2x - cx, p2y - cy)
    dx, dy = (x2 - x1), (y2 - y1)
    dr = (dx ** 2 + dy ** 2)**.5
    big_d = x1 * y2 - x2 * y1
    discriminant = circle_radius ** 2 * dr ** 2 - big_d ** 2
    if discriminant < 0:  # No intersection between circle and line
        
        return []
    else:  # There may be 0, 1, or 2 intersections with the segment
        intersections = [
            (cx + (big_d * dy + sign * (-1 if dy < 0 else 1) * dx * discriminant**.5) / dr ** 2,
             cy + (-big_d * dx + sign * abs(dy) * discriminant**.5) / dr ** 2)
            for sign in ((1, -1) if dy < 0 else (-1, 1))]  # This makes sure the order along the segment is correct
        if not full_line:  # If only considering the segment, filter out intersections that do not fall within the segment
            fraction_along_segment = [(xi - p1x) / dx if abs(dx) > abs(dy) else (yi - p1y) / dy for xi, yi in intersections]
            intersections = [pt for pt, frac in zip(intersections, fraction_along_segment) if 0 <= frac <= 1]
        if len(intersections) == 2 and abs(discriminant) <= tangent_tol:  # If line is tangent to circle, return just one point (as both intersections have same location)
            return [intersections[0]]
        else:
            return intersections

def Find_distance_matrix(N,i):
    Pos_vec = [(r_x[num,i], r_y[num,i]) for num in range(N)]
    Dist_mat = distance.cdist(Pos_vec,Pos_vec, metric = 'euclidean')
         
    return Dist_mat

def Angle(vec): return np.arctan2(vec[1],np.dot(vec,[1,0]))

def Reflected_vector(r_old,  phi, dx, dy, r_box):
    r_new = r_old + np.array([dx,dy])
    phi_new = phi
    while norm(r_new) > r_box:
        intercepts = circle_line_segment_intersection((0,0),r_box, (r_old[0], r_old[1]), (r_new[0], r_new[1]),full_line = False)
        
        intercepts_vec =  np.array([intercepts[0][0], intercepts[0][1]])
        a = np.array([r_new[0] -intercepts_vec[0], r_new[1] - intercepts_vec[1]])
        r_new = r_new - 2*(np.dot(intercepts_vec,a) / norm(intercepts_vec) ** 2) * intercepts_vec
        r_old = intercepts_vec
            
        a_reflected = a - 2*(np.dot(intercepts_vec,a) / norm(intercepts_vec) ** 2) * intercepts_vec
        phi_new = Angle(a_reflected)

    return r_new[0], r_new[1], phi_new

def Reflective_Boundary(r_x,r_y, phi_i,dx,dy, r_box):
    rx_new,ry_new, phi_new = Reflected_vector(np.array([r_x,r_y]),phi_i, dx,dy,r_box)
    return rx_new,ry_new, phi_new 

def Confined_abp_move(t,dt,N,r_b):
    for i in range(0, int(t/dt)-1): # time evolution
        # Find position of each particles at time i
        Dist_mat = Find_distance_matrix(N,i)
        
        # Run interactions
        for p1 in range(N):
            r_dx_new_p1, r_dy_new_p1, dphi_new_p1 = 0,0,np.sqrt(2*Dr*dt) * np.random.randn()    # initializing increment for r_x, r_y, phi for i'th particle
           
            for p2 in range(N):
                if p1 == p2:
                    continue
                dist = Dist_mat[p1,p2]
                # Greater than interaction range then don't interact
                if dist > epsilon_int:
                    continue
                
                ri_x = r_x[p1,i]
                ri_y = r_y[p1,i]
                rj_x = r_x[p2,i]
                rj_y = r_y[p2,i]    # (x,y) of vectors ri and rj
                
                # Positional repulsion
                if dist  <= zeta_ex: 
                    r_dx_new_p1 = r_dx_new_p1 - k_pos * (rj_x - ri_x) * dt
                    r_dy_new_p1 = r_dy_new_p1 - k_pos * (rj_y - ri_y) * dt
                    
                # Angular repulsion
                if dist <= epsilon_ex:
                    alpha_ji = Angle([rj_x - ri_x, rj_y - ri_y])
                    
                    dphi_new_p1 = dphi_new_p1 - k_rad * np.sin(alpha_ji - phi[p1,i]) * dt
                # Alignment     
                if dist <= epsilon_aa:
                    dphi_new_p1 = dphi_new_p1 + mu_plus*(1-(dist/epsilon_aa)**2) * np.sin(phi[p2,i] - phi[p1,i])*dt
                # Anti-alignment
                if dist > epsilon_aa: 
                    dphi_new_p1 = dphi_new_p1 - mu_minus*(4*(dist-epsilon_aa)*(epsilon_int-dist)) / (epsilon_int-epsilon_aa)**2 * \
                                        np.sin(phi[p2,i] - phi[p1,i]) *dt
                                   
            # Time to add increments for p1 from interaction of all other particles
            dx = v * np.cos(phi[p1,i]) * dt + r_dx_new_p1
            dy = v * np.sin(phi[p1,i]) * dt + r_dy_new_p1
            
            # Check if p1's future position is outside of the boundary
            r_x[p1,i+1], r_y[p1,i+1], phi[p1,i+1]  = \
                       Reflective_Boundary(r_x[p1,i], r_y[p1,i], phi[p1,i] + dphi_new_p1, dx, dy, box_r)
       
    
        print('Time step: ',i)
# CONSTANTS
v = 37                         # swimming speed of B. Subtilis [m/s]
Dr = 0.3 #   1                    # rotational diffusion coefficient of B. Subtilis

epsilon_int = 30  # 1                  # beyound which interaction will be gone
epsilon_aa = 20 #    0.2             # allignment & antialignment transition radius
epsilon_ex = 6 #     0.1              # angular exclusion radius
zeta_ex =   3.5    # 0                 # positional exclusion radius

k_rad = 5   #      10                # anuglar exclusion strength
k_pos = 0.02  #    0                 # positional exclusion strength


# ADJUSTABLE PARAMETERS
t = 20        # time over which motion is observed [s]
dt = 0.01     # time step between recorded positions
N = 142 #2000          # number of cells 

mu_plus = 0.3  #4#  # alignment strength
mu_minus = 0.1  #0.004   # anti-alignment strength

# Boundary box
box_r = 30       # boundary circle radius


# Packing fraction & density (Grossman et al. 2014 PRL)
psi = N * np.pi * epsilon_ex ** 2 / (np.pi* box_r **2)
rho = N / (np.pi* box_r **2)

# INITIALIZING VARIABLES
theta = np.zeros((N,1))  # initial polar angle [radians]
r = np.zeros((N,int(t/dt)))  # initial radial position [m]
r_x = np.zeros((N,int(t/dt)))
r_y = np.zeros((N,int(t/dt)))
r_x_buffer = np.zeros((N,int(t/dt)))
r_y_buffer = np.zeros((N,int(t/dt)))
phi = np.zeros((N,int(t/dt)))  # speed angle


# Initializing x y theta; vx vy will be initialized in ABP move
for n in range(N):  
    # x positions
    r[n,0] = np.random.uniform(0,box_r-5)
    theta[n] = np.random.uniform(-2*np.pi, 2*np.pi)
    phi[n,0] = np.random.uniform(-2*np.pi, 2*np.pi)
    r_x[n,0] =  r[n,0]*np.cos(theta[n])  
    r_y[n,0] =  r[n,0]*np.sin(theta[n])  
    
Confined_abp_move(t,dt,N,box_r)   

print("Packing Fraction = ", psi)
print("Density = ", rho)

for  i in range(0,1000):
    fig = plt.figure(dpi=111)
    ax = plt.axes(xlim=(-1*box_r - 4, 1*box_r + 4), ylim=(-1*box_r- 4, 1*box_r + 4))
    ax.set_aspect(1)
    circle2 = plt.Circle((0, 0), box_r, color='k', fill=False)
    ax.add_patch(circle2)
    ax.quiver(r_x[:,i], r_y[:,i],\
                  5 * np.cos(phi[:,i]), 5 * np.sin(phi[:,i]), scale_units='x',scale = 1)
    ax = plt.gca()
    plt.axis('off')
    plt.savefig(r"C:\Users\Kaneki\Documents\test\snap_{0}.png".format(i))
    plt.close()             
