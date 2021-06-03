import tensorflow as tf
import math as math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import mpl_toolkits.mplot3d.axes3d as p3
import numpy as np
# particle tensor:

#[[x,y,z],[x',y',z'],[q,m,some other property]]


#constants

Mu0 = 4*math.pi*10**(-7)
Epsilon0 = 8.854187812813*10**(-12)
delta_t = 0.001
#units

Simulation_Data = []
particles = tf.convert_to_tensor([tf.convert_to_tensor([[1.0,1.0,0.0],[1.0,2.0,0.0],[10.0,10.0,10.0]]),tf.convert_to_tensor([[1.0,2.0,0.0],[6.0,0.0,0.0],[10.0,10.0,10.0]])])

def position_constructor(position):
    out = []
    for i in range(particles.shape[0]):
        out.append(tf.convert_to_tensor(position))
    return tf.convert_to_tensor(out)


def Coloumb_Potential(args):
    k = tf.convert_to_tensor([1/(Epsilon0*4*math.pi)])
    particle = args[0]
    position = args[1]
    return tf.cond(tf.reduce_all(particle[0]==position),lambda: tf.convert_to_tensor([0.0,0.0,0.0]),lambda:(k*particle[2][0]/tf.norm((position-particle[0]))**3)*(position-particle[0]))


def Electric_Field(pos):
    return tf.reduce_sum(tf.vectorized_map(Coloumb_Potential, (particles,position_constructor(pos))),0)

#bio savat law for moving charge
def Magneto_Potential(args):
    particle = args[0]
    position = args[1]
    return tf.cond(tf.reduce_all(particle[0]==position),lambda: tf.convert_to_tensor([0.0,0.0,0.0]),lambda: ((particle[2][0]*Mu0/(4*math.pi*tf.norm(position-particle[0])**3))*tf.linalg.cross(particle[1],(position-particle[0]))))


def Magnetic_Field(pos):
    return tf.reduce_sum(tf.vectorized_map(Magneto_Potential, (particles,position_constructor(pos))),0)


def Lorentz_Force(particle):
    E = Electric_Field(particle[0])
    B = Magnetic_Field(particle[0])
    return particle[2][0]*(E+tf.linalg.cross(particle[1],B))


def Particle_Timestep(particle):
    a = Lorentz_Force(particle)/particle[2][1]
    position = tf.add(particle[1]*delta_t,particle[0])
    velocity = tf.add(a*delta_t,particle[1])
    state = particle[2]
    return tf.convert_to_tensor([position,velocity,state])

def Timestep():
    return tf.map_fn(Particle_Timestep,particles)#this dont work with vectorized map

#this can be changed for memory efficeny

def animate(State):
    vis._offsets3d = ([State[0:Simulation_Data.shape[1],0,0], State[0:Simulation_Data.shape[1],0,1], State[0:Simulation_Data.shape[1],0,2]])





#Logic loop
Simulation_Data.append(particles)
for i in range(10):
    particles = Timestep()
    Simulation_Data.append(particles)

Simulation_Data = np.array(Simulation_Data)
print(Simulation_Data)

plt.style.use('dark_background')
fig = plt.figure()
ax = p3.Axes3D(fig)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_xlim3d(-10000, 10000)
ax.set_ylim3d(-10000,10000)
ax.set_zlim3d(-10000,10000)
vis = ax.scatter(Simulation_Data[0,0:Simulation_Data.shape[1],0,0], Simulation_Data[0,0:Simulation_Data.shape[1],0,1], Simulation_Data[0,0:Simulation_Data.shape[1],0,2],color=(1,0,0,1))
ani = animation.FuncAnimation(fig, animate, frames=Simulation_Data, interval=30, repeat=True)
# ani.save('out.gif',writer="ffmpeg")
plt.show()


#visualisation/ map coordinate tensors with field functions?
