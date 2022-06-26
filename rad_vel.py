from matplotlib import projections
import numpy as np
import matplotlib.pyplot as plt
import math

def Acc( pos, mass, G, softening ):
    x = pos[:,0:1]
    y = pos[:,1:2]
    z = pos[:,2:3]  

    x_diff= (x[0,0]-x[1,0])
    y_diff= (y[0,0]-y[1,0])
    z_diff= (z[0,0]-z[1,0])
    diff_vec=np.array([x_diff,y_diff,z_diff])

    inv_r3 = (x_diff**2+y_diff**2+z_diff**2 + softening**2)**(-1.5)
    a=np.zeros((2,3))
    a[0]=-G*mass[1]*inv_r3*diff_vec
    a[1]=G*mass[0]*inv_r3*diff_vec  
    return a


N         = 2 
t         = 0  
dt        = 0.01   
softening = 0   
G         = 1.0

plotRealTime = True # Animation Switch

# Orbital Elements

a=5
e=0.5
f=(0)*(np.pi/180)
i=(45)*(np.pi/180)
raan=(45)*(np.pi/180)
w=(90)*(np.pi/180)
mass1=100
mass2=200

p=a*(1-e**2)
r=p/(1+e*math.cos(f))
u=G*(mass1+mass2)
h=(p*u)**0.5

tEnd = 4*2*np.pi*math.sqrt((a**3)/u) 

mat1=np.array(([math.cos(raan),-math.sin(raan),0],[math.sin(raan),math.cos(raan),0],[0,0,1]))
mat2=np.array(([1,0,0],[0,math.cos(i),-math.sin(i)],[0,math.sin(i),math.cos(i)]))
mat3=np.array(([math.cos(w),-math.sin(w),0],[math.sin(w),math.cos(w),0],[0,0,1]))
rot_mat=(mat1 @ mat2) @ mat3
r_pqw=np.array(([r*math.cos(f)],[r*math.sin(f)],[0]))
v_pqw=np.array(([-((u/p)**0.5)*math.sin(f)],[((u/p)**0.5)*(e+math.cos(f))],[0]))
r_vec=rot_mat @ r_pqw
v_vec=rot_mat @ v_pqw
r_vec=r_vec.reshape(3)
v_vec=v_vec.reshape(3)
r1_vec=-(mass2/(mass1+mass2))*r_vec
r2_vec=(mass1/(mass1+mass2))*r_vec

v1_vec=-(mass2/(mass1+mass2))*v_vec
v2_vec=(mass1/(mass1+mass2))*v_vec

pos1=np.array([r1_vec[0],r1_vec[1],r1_vec[2]])
pos2=np.array([r2_vec[0],r2_vec[1],r2_vec[2]])
vel1=np.array([v1_vec[0],v1_vec[1],v1_vec[2]])
vel2=np.array([v2_vec[0],v2_vec[1],v2_vec[2]])

m = np.array([mass1,mass2])
mass = np.reshape(m,(2,1))
pos  = np.array((pos1,pos2))
vel  = np.array((vel1,vel2))

vel -= np.mean(mass * vel,0) / np.mean(mass)

acc = Acc( pos, mass, G, softening )
Nt = int(np.ceil(tEnd/dt))
plnt1=np.zeros((Nt,3))
plnt2=np.zeros((Nt,3))
curve1=np.zeros((Nt,2))
curve2=np.zeros((Nt,2))

pos_trail = np.zeros((N,3,Nt+1))
pos_trail[:,:,0] = pos
curve1_trail = np.zeros(Nt+1)
curve1_trail[0] = vel1[0]
curve2_trail = np.zeros(Nt+1)
curve2_trail[0] = vel2[0]
t_all = np.arange(Nt+1)*dt

fig = plt.figure(figsize=(10,10), dpi=80)
grid = plt.GridSpec(3, 1, wspace=0.0, hspace=0.3)
ax1 = plt.subplot(grid[0:2,0],projection="3d")
ax2 = plt.subplot(grid[2,0])

for i in range(Nt):
    vel += acc * dt/2.0
    pos += vel * dt
    acc = Acc( pos, mass, G, softening )
    vel += acc * dt/2.0
    t += dt
    pos_trail[:,:,i+1] = pos
    curve1_trail[i+1] = vel[0,0]
    curve2_trail[i+1] = vel[1,0]

    if plotRealTime :
        plt.sca(ax1)
        plt.cla()
        xx1 = pos_trail[0,0,0:i+1]
        yy1 = pos_trail[0,1,0:i+1]
        zz1 = pos_trail[0,2,0:i+1]
        xx2 = pos_trail[1,0,0:i+1]
        yy2 = pos_trail[1,1,0:i+1]
        zz2 = pos_trail[1,2,0:i+1]
        ax1.scatter(xx1,yy1,zz1,s=1,color='r')
        ax1.scatter(xx2,yy2,zz2,s=1,color='b')
        ax1.scatter(pos[0,0],pos[0,1],pos[0,2],s=50,color='red')
        ax1.scatter(pos[1,0],pos[1,1],pos[1,2],s=50,color='blue')
        ax1.set(xlim=(-5, 5), ylim=(-5, 5), zlim=(-5,5))

        plt.sca(ax2)
        plt.cla()
        ax1.set_xlabel('\n\nX-axis\n             Observation --------->', fontsize=13)
        ax1.set_ylabel('Y-axis', fontsize=13)
        ax1.set_zlabel('Z-axis', fontsize=13)
        plt.scatter(t_all,curve1_trail,color='red',s=1)
        plt.scatter(t_all[i],curve1_trail[i],color='red',s=50,label='Radial Velocity of Star 1')
        plt.scatter(t_all,curve2_trail,color='blue',s=1)
        plt.scatter(t_all[i],curve2_trail[i],color='blue',s=50,label='Radial Velocity of Star 2')
        # plt.grid()
        plt.legend(loc='upper right',bbox_to_anchor=(1.01,1.35),fontsize=13)
        ax2.set(xlim=(0, tEnd))
        ax2.set_xlabel('Time',fontsize=13)
        ax2.set_ylabel('Radial Velocity',fontsize=13)
        plt.pause(0.0001)
    else:
        plnt1[i,:]=np.array([pos[0,:]])
        plnt2[i,:]=np.array([pos[1,:]])
        curve1[i,:]=np.array((t,vel[0,0]))
        curve2[i,:]=np.array((t,vel[1,0]))


if not(plotRealTime):
    plt.sca(ax1)
    ax1.set_xlabel('\n\nX-axis\n             Observation --------->', fontsize=13)
    ax1.set_ylabel('Y-axis', fontsize=13)
    ax1.set_zlabel('Z-axis', fontsize=13)
    ax1.scatter(plnt1[:,0],plnt1[:,1],plnt1[:,2],c='r',s=3)
    ax1.scatter(plnt1[0,0],plnt1[0,1],plnt1[0,2],c='r',s=50)
    ax1.scatter(plnt2[:,0],plnt2[:,1],plnt2[:,2],c='b',s=3)
    ax1.scatter(plnt2[0,0],plnt2[0,1],plnt2[0,2],c='b',s=50)
    plt.sca(ax2)
    plt.scatter(curve1[:,0],curve1[:,1],c='r',s=3,label='Radial Velocity of Star 1')
    plt.scatter(curve2[:,0],curve2[:,1],c='b',s=3,label='Radial Velocity of Star 2')
    ax2.set_xlabel('Time',fontsize=13)
    ax2.set_ylabel('Radial Velocity',fontsize=13)
    plt.grid()
    plt.legend(loc='upper right',bbox_to_anchor=(1.01,1.35),fontsize=13)
plt.show()