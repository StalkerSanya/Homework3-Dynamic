# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 15:06:26 2019

@author: Stalker
"""

import sympy as sm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
import math as m
import numpy as np

### TASK3 Finding Jacobian
def rotx(q):
    r = sm.rot_axis1(-q)
    return r
def roty(q):
    r = sm.rot_axis2(-q)
    return r
def rotz(q):
    r = sm.rot_axis3(-q)
    return r

q1, q2, d3 = sm.symbols("q1 q2 d3")
d1, a2 = sm.symbols("d1 a2")

Px03 = (a2 + d3)*sm.cos(q2)*sm.cos(q1)
Py03 = (a2 + d3)*sm.cos(q2)*sm.sin(q1)
Pz03 = d1 + (a2 + d3)*sm.sin(q2)

Px13 = (a2 + d3)*sm.cos(q2)*sm.cos(q1)
Py13 = (a2 + d3)*sm.cos(q2)*sm.sin(q1)
Pz13 = (a2 + d3)*sm.sin(q2)

Px23 = (a2 + d3)*sm.cos(q2)*sm.cos(q1)
Py23 = (a2 + d3)*sm.cos(q2)*sm.sin(q1)
Pz23 = (a2 + d3)*sm.sin(q2)

J_classic = sm.Matrix([[sm.diff(Px03, q1), sm.diff(Px03, q2), sm.diff(Px03, d3)],
                       [sm.diff(Py03, q1), sm.diff(Py03, q2), sm.diff(Py03, d3)],
                       [sm.diff(Pz03, q1), sm.diff(Pz03, q2), sm.diff(Pz03, d3)]])
    
print("Classical aproach gives us Jacobian:")    
print(J_classic, "\n")

print("Geometric aproach gives us Jacobian:")
o03 = sm.Matrix([[Px03],[Py03],[Pz03]])
z0 = sm.Matrix([[0],[0],[1]])
J_cross1 = z0.cross(o03)
print("Column1: ")
print(J_cross1)

o13 = sm.Matrix([[Px13],[Py13],[Pz13]])
z1 = rotz(q1)*rotx(sm.pi/2)*sm.Matrix([[0],[0],[1]])
J_cross2 = z1.cross(o13)
print("Column2: ")
print(J_cross2)

print(rotz(sm.pi/4)*rotz(sm.pi/2)*rotx(sm.pi/2)*rotz(0)*roty(sm.pi/2)*rotz(sm.pi/2)*sm.Matrix([[0],[0],[1]]))
z2 = rotz(q1)*rotx(sm.pi/2)*rotz(q2)*roty(sm.pi/2)*rotz(sm.pi/2)*sm.Matrix([[0],[0],[1]])
#print(z1)
J_cross3 = z2
print("Column3: ")
print(J_cross3, "\n")



### TASK 5 Finding linear velocity

t = np.linspace( 0, 4*np.pi, 800 )
a2_ = 1
q1_ = np.sin(t)
q2_ = np.cos(2*t)
d3_ = np.sin(3*t)
dq1 = np.cos(t)
dq2 = -2*np.sin(2*t)
dd3 = 3*np.cos(3*t)
Jc = np.array([[-(a2_ + d3_)*np.sin(q1_)*np.cos(q2_), -(a2_ + d3_)*np.sin(q2_)*np.cos(q1_), np.cos(q1_)*np.cos(q2_)], 
             [(a2_ + d3_)*np.cos(q1_)*np.cos(q2_), -(a2_ + d3_)*np.sin(q1_)*np.sin(q2_), np.sin(q1_)*np.cos(q2_)], 
             [0, (a2_ + d3_)*np.cos(q2_), np.sin(q2_)]])
dPx = Jc[0,0]*dq1 + Jc[0,1]*dq2 + Jc[0,2]*dd3
dPy = Jc[1,0]*dq1 + Jc[1,1]*dq2 + Jc[1,2]*dd3
dPz = Jc[2,0]*dq1 + Jc[2,1]*dq2 + Jc[2,2]*dd3

fig = plt.figure(1)
plt.title("Projection of velocity on X direction")
plt.scatter(t, dPx, s = 5)
plt.xlabel("t")
plt.ylabel("dPx")
plt.show

fig = plt.figure(2)
plt.title("Projection of velocity on Y direction")
plt.scatter(t, dPy, s = 5)
plt.xlabel("t")
plt.ylabel("dPy")
plt.show

fig = plt.figure(3)
plt.title("Projection of velocity on Z direction")
plt.scatter(t, dPz, s = 5)
plt.xlabel("t")
plt.ylabel("dPz")
plt.show


fig = plt.figure(4)
axes = Axes3D(fig)
axes.scatter3D(dPx, dPy, dPz, s = 2, c = "green")
axes.set_xlabel('dPx, (cm)')
axes.set_ylabel('dPy, (cm)')
axes.set_zlabel('dPz, (cm)')
plt.show()



### TASK 6 Finding joint trajectory  
## By using IK
t1 = np.linspace( 0, 4*np.pi, 800 )
a2_ = 1
d1_ = 1
#Px03 = (a2 + d3)*sm.cos(q2)*sm.cos(q1)
#Py03 = (a2 + d3)*sm.cos(q2)*sm.sin(q1)
#Pz03 = d1 + (a2 + d3)*sm.sin(q2)
Px_t = 2*a2_*np.sin(t)
Py_t = 2*a2_*np.cos(2*t)
Pz_t = d1_*np.sin(3*t)
q1_t = np.arctan2(Py_t, Px_t)
q2_t = np.arctan2(Pz_t - d1_, np.sqrt(Px_t**2 + Py_t**2))
d3_t = np.sqrt(Px_t**2 + Py_t**2) - a2_

fig = plt.figure(5)
plt.title("Joint trajectory for q1")
plt.scatter(t, q1_t, s = 5, label = "with IK solution")
plt.xlabel("t")
plt.ylabel("q1_t")
plt.legend()
plt.show

fig = plt.figure(6)
plt.title("Joint trajectory for q2")
plt.scatter(t, q2_t, s = 5, label = "with IK solution")
plt.xlabel("t")
plt.ylabel("q2_t")
plt.legend()
plt.show

fig = plt.figure(7)
plt.title("Joint trajectory for d3")
plt.scatter(t, d3_t, s = 5, label = "with IK solution")
plt.xlabel("t")
plt.ylabel("q3_t")
plt.legend()
plt.show


## By using differential kinematics aproach

#t = np.linspace( 0, 4*np.pi, 800 )
t = np.linspace( 0, 4*np.pi, 800 )
a2_ = 1
d1_ = 1
Px_t = 2*a2_*np.sin(t)
Py_t = 2*a2_*np.cos(2*t)
Pz_t = d1_*np.sin(3*t)
dPx_t = 2*a2_*np.cos(t)
dPy_t = -4*a2_*np.sin(2*t)
dPz_t = 3*d1_*np.cos(3*t)
q1t = np.zeros((t.shape[0],), dtype = "float32")
q2t = np.zeros((t.shape[0],), dtype = "float32")
d3t = np.zeros((t.shape[0],), dtype = "float32")
Jc_inv = []
q1t[0] = 0
q2t[0] = 0
d3t[0] = 0
for i in range(1, t.shape[0]):
    Jt = np.array([[-(a2_ + d3t[i-1])*np.sin(q1t[i-1])*np.cos(q2t[i-1]), -(a2_ + d3t[i-1])*np.sin(q2t[i-1])*np.cos(q1t[i-1]), np.cos(q1t[i-1])*np.cos(q2t[i-1])], 
                   [(a2_ + d3t[i-1])*np.cos(q1t[i-1])*np.cos(q2t[i-1]), -(a2_ + d3t[i-1])*np.sin(q1t[i-1])*np.sin(q2t[i-1]), np.sin(q1t[i-1])*np.cos(q2t[i-1])], 
                   [0, (a2_ + d3t[i-1])*np.cos(q2t[i-1]), np.sin(q2t[i-1])]])
    Jc_inv.append(np.linalg.inv(Jt))
    q1t[i] = q1t[i-1] + (Jc_inv[i-1][0,0]*dPx_t[i-1] + Jc_inv[i-1][0,1]*dPy_t[i-1] +Jc_inv[i-1][0,2]*dPz_t[i-1])*(t[i]-t[i-1])
    q2t[i] = q2t[i-1] + (Jc_inv[i-1][1,0]*dPx_t[i-1] + Jc_inv[i-1][1,1]*dPy_t[i-1] +Jc_inv[i-1][1,2]*dPz_t[i-1])*(t[i]-t[i-1])
    d3t[i] = d3t[i-1] + (Jc_inv[i-1][2,0]*dPx_t[i-1] + Jc_inv[i-1][2,1]*dPy_t[i-1] +Jc_inv[i-1][2,2]*dPz_t[i-1])*(t[i]-t[i-1])


          
fig = plt.figure(5)
plt.title("Joint trajectory for q1")
plt.scatter(t, q1t, s = 5, label = "with diff. kinem. aproach")
plt.legend()
plt.xlabel("t")
plt.ylabel("q1t")
plt.show

fig = plt.figure(6)
plt.title("Joint trajectory for q2")
plt.scatter(t, q2t, s = 5, label = "with diff. kinem. aproach")
plt.xlabel("t")
plt.ylabel("q2t")
plt.legend()
plt.show

fig = plt.figure(7)
plt.title("Joint trajectory for d3")
plt.scatter(t, d3t, s = 5, label = "with diff. kinem. aproach")
plt.xlabel("t")
plt.ylabel("q3t")
plt.legend()
plt.show  

          














