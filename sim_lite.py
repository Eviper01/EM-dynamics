import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation
import mpl_toolkits.mplot3d.axes3d as ax3d
import numpy as np
from matplotlib.patches import Circle
import pandas as pd

class Particle(object):
    """A particle that carries charge and will radiate an electric field."""
    def __init__(self, position, velocity, charge):
        self.position = position
        self.velocity = velocity
        self.charge = charge
    def tostring(self):
        print("pos = {x}, vel = {dx}, charge = {q}".format(x=self.position, dx=self.velocity, q=self.charge))
    def columb_potential(self):
        potential = np.where(np.sqrt((xv-self.position[0])**2 + (yv-self.position[1])**2) < 1e-6, 0 , (k*self.charge)/(np.sqrt((xv-self.position[0])**2 + (yv-self.position[1])**2)))
        return potential

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

#constants
Mu0 = 4*np.pi*10**(-7)
Epsilon0 = 8.854187812813*10**(-12)
delta_t = 0.001
k = 1/(4*np.pi*Epsilon0)

#canvas setup
n = 5000
size = 400
x = np.linspace(-size, size, n)
y = np.linspace(-size, size, n)
xv, yv = np.meshgrid(x, y, sparse=False)

def main():
    p1 = Particle([20, 10], [0,0], +1)
    p2 = Particle([-30, 20], [0,0], -1)
    p3 = Particle([0, 0], [0,0], -5)
    p4 = Particle([20, -30], [0,0], +3)
    particles = [p1, p2, p3, p4]
    Ex, Ey, Vfield = Efield(particles)
    field_visualise(Ex, Ey, particles).show()
main()
    #deprciated
#
# plt.pcolormesh(xv, yv, (Vfield), cmap = cm.bwr)
# plt.show()
