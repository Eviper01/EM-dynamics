import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation
import mpl_toolkits.mplot3d.axes3d as ax3d
import numpy as np
from matplotlib.patches import Circle
import pandas as pd
from matplotlib.animation import FuncAnimation


class Particle(object):
    """A particle that carries charge and will radiate an electric field."""
    def __init__(self, position, velocity, charge, mass):
        self.position = position
        self.velocity = velocity
        self.charge = charge
        self.mass = mass
    def tostring(self):
        print("pos = {x}, vel = {dx}, charge = {q}".format(x=self.position, dx=self.velocity, q=self.charge))
    def columb_potential(self):
        potential = (k*self.charge)/(np.sqrt((xv-self.position[0])**2 + (yv-self.position[1])**2))
        return potential
    def travel(self):
        self.position += self.velocity*delta_t
    def accelerate(self, Ex, Ey):
        qEy, qEx = np.gradient(self.columb_potential())
        Ex += qEx
        Ey += qEy
        try:
            self.velocity[0] += Ex[self.pindex()[1], self.pindex()[0]]*self.charge/self.mass
            self.velocity[1] += Ey[self.pindex()[1], self.pindex()[0]]*self.charge/self.mass
        except IndexError:
            pass
    def pindex(self):
        idx = (np.abs(x - self.position[0])).argmin()
        idy = (np.abs(y - self.position[1])).argmin()
        return idx, idy
    def get_summary(self):
        summary = [self.position, self.charge]
        return summary


def Efield(particles):
    Vfield = particles[0].columb_potential()
    for particle in particles[1:len(particles)]:
        Vfield += particle.columb_potential()
    Ey, Ex = np.gradient(Vfield)
    return -Ex, -Ey, Vfield


def field_visualise(Ex, Ey, particles):
#plots field lines
    color = 2 * np.log(np.hypot(Ex, Ey))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.streamplot(xv, yv, Ex, Ey, color=color, linewidth=1, cmap=cm.plasma,
                  density=2, arrowstyle='->', arrowsize=1.5)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_xlim(-size,size)
    ax.set_ylim(-size,size)
    ax.set_aspect('equal')
    #plots charges
    charge_colors = {True: '#aa0000', False: '#0000aa'}
    for particle in particles:
        ax.add_artist(Circle(particle.position, size*0.05, color=charge_colors[particle.charge>0]))

    return plt

def animated_field(field):
#plots field lines
    Ex = field[0]
    Ey = field[1]
    color = 2 * np.log(np.hypot(Ex, Ey))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.streamplot(xv, yv, Ex, Ey, color=color, linewidth=1, cmap=cm.plasma,
                  density=2, arrowstyle='->', arrowsize=1.5)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_xlim(-size,size)
    ax.set_ylim(-size,size)
    ax.set_aspect('equal')
    return plt


#constants
# Mu0 = 4*np.pi*10**(-7)
Mu0 = 1
dim = 2
# Epsilon0 = 8.854187812813*10**(-12)
delta_t = 1
# k = 1/(4*np.pi*Epsilon0)
k = 1e2
#canvas setup
n = 500
size = 80
x = np.linspace(-size, size, n)
y = np.linspace(-size, size, n)
xv, yv = np.meshgrid(x, y, sparse=False)
# data arhcitercrue
# list constiing of the [field, [each particles]]
data = []
def main():

    p1 = Particle(np.array([20.0, 10.0]), np.array([0.0,0.0]), +1, 1)
    p2 = Particle(np.array([-40.0, 20.0]), np.array([0.0,0.0]), -1, 1)
    p3 = Particle(np.array([0.0, 0.0]), np.array([0.0,0.0]), -5, 1)
    p4 = Particle(np.array([20.0, -30.0]), np.array([0.0,0.0]), +3, 1)
    particles = [p1 ,p2, p3, p4]
    # particles = [p1, p2]

    for i in range(100):
        Ex, Ey, Vfield = Efield(particles)
        data.append([Ex, Ey])
        # field_visualise(Ex, Ey, particles).show()
        for particle in particles:
            particle.accelerate(Ex, Ey)
            particle.travel()

    fig, ax = plt.subplots()
    xdata, ydata = [], []
    ln, = plt.plot([], [], 'ro')

    def init():
        ax.set_xlim(-size, size)
        ax.set_ylim(-size, size)
        return ln,

    def update(frame):
        xdata.append(frame)
        ydata.append(np.sin(frame))
        ln.set_data(xdata, ydata)
        return ln,

    ani = FuncAnimation(fig, update, frames=data),
                        init_func=init, blit=True)
    plt.show()


main()

#add charge anihilation
#animation

    #deprciated
#
# plt.pcolormesh(xv, yv, (Vfield), cmap = cm.bwr)
# plt.show()
