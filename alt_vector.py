import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation
import numpy as np
from matplotlib.patches import Circle

class Particle():
    """A particle that carries charge and will radiate an electric field."""
    def __init__(self, position, velocity, charge, mass):
        global Particle_id
        self.position = position
        self.velocity = velocity
        self.charge = charge
        self.mass = mass
        self.dead = False
        self.id = Particle_id
        Particle_id += 1
    def tostring(self):
        print("pos = {x}, vel = {dx}, charge = {q}".format(x=self.position, dx=self.velocity, q=self.charge))
    def columb_potential(self):
        potential = (k*self.charge)/(np.sqrt((xv-self.position[0])**2 + (yv-self.position[1])**2))
        return potential
    def travel(self):
        self.position += self.velocity*delta_t
    def accelerate(self, Ex, Ey):
        qEx, qEy = np.gradient(self.columb_potential())
        nEx = Ex + qEx
        nEy = Ey + qEy
        try:
            self.velocity[0] += nEx[self.pindex()[1], self.pindex()[0]]*self.charge/self.mass
            self.velocity[1] += nEy[self.pindex()[1], self.pindex()[0]]*self.charge/self.mass
        except IndexError:
            pass
    def check_collision(self, particles):
        for particle in particles:
            if np.linalg.norm(particle.position - self.position) < 10 and particle != self:
                self.dead = True
                new_charge = self.charge + particle.charge
                new_mass = self.mass + particle.mass
                new_velocity = (self.velocity*self.mass + particle.velocity*particle.mass)/new_mass
                new_position = (particle.position + self.position)/2
                queue.append(np.array([new_position,new_velocity, new_charge, new_mass, {self.id, particle.id}]))
    def pindex(self):
        idx = (np.abs(x - self.position[0])).argmin()
        idy = (np.abs(y - self.position[1])).argmin()
        return idx, idy
    def get_summary(self):
        summary = [self.position*1, self.charge*1]
        return summary[:]


def Efield(particles):
    Vfield = particles[0].columb_potential()
    for particle in particles[1:len(particles)]:
        Vfield += particle.columb_potential()
    Ex, Ey = np.gradient(Vfield)
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

def remove_dupe(dupelist):
    col_ids = [array[4] for array in dupelist]
    out_ids = []
    for col_id in col_ids:
        if col_id not in out_ids:
            out_ids.append(col_id)
    out = []
    for element in dupelist:
        if element[4] in out_ids:
            out.append(element)
            out_ids.remove(element[4])
    return out


#constants
# Mu0 = 4*np.pi*10**(-7)
Mu0 = 1
dim = 2
# Epsilon0 = 8.854187812813*10**(-12)
delta_t = 1
# k = 1/(4*np.pi*Epsilon0)
k = 1e2
#canvas setup
n = 1000
size = 160
steps = 50
x = np.linspace(-size, size, n)
y = np.linspace(-size, size, n)
xv, yv = np.meshgrid(x, y, sparse=False)
# data arhcitercrue
# list constiing of the [field, [each particles]]
Particle_id = 0
data = []
queue = []


def main():
    global queue
    particles = []
    # p1 = Particle(np.array([20.0, 10.0]), np.array([0.0,0.0]), +1, 1)
    # p2 = Particle(np.array([-40.0, 20.0]), np.array([0.0,0.0]), -1, 1)
    # p3 = Particle(np.array([0.0, 0.0]), np.array([0.0,0.0]), -1, 1)
    # p4 = Particle(np.array([20.0, 0.0]), np.array([0.0, 2.0]), +1, 1)
    # particles = [p3, p4]
    for i in range(5):
        px = 100*np.random.random()-50
        py = 100*np.random.random()-50
        q = 10*np.random.random()-5
        particles.append(Particle(np.array([px,py]), np.array([0.0,0.0]), q, 1))


    for i in range(steps):

        queue = []
        charge_data = []
        Ex, Ey, Vfield = Efield(particles)
        # field_visualise(Ex, Ey, particles).show()
        for particle in particles:
            particle.accelerate(Ex, Ey)
        for particle in particles:
            particle.travel()
        for particle in particles:
            particle.check_collision(particles)
        for particle in particles:
            charge_data.append(particle.get_summary())
        new_p = []
        for particle in particles:
            if particle.dead is False:
                new_p.append(particle)
        particles = new_p
        queue = remove_dupe(queue)  # not having this is fucking funny
        for particle in queue:
            particles.append(Particle(particle[0], particle[1], particle[2], particle[3]))
        data.append(([Ex, Ey], charge_data))

    print("Simulation Complete")
    #animatin method

    def animate(frame):
        ax.collections = [] # clear lines streamplot
        ax.patches = [] # clear arrowheads streamplot
        ax.artists = [] #clears the particles postions
        color = 2 * np.log(np.hypot(Ex, Ey))
        stream = ax.streamplot(xv,yv,frame[0][0], frame[0][1], color=color, density=2, linewidth=1, cmap=cm.plasma, arrowsize=1.5, arrowstyle='->')
        dots = []
        charge_colors = {True: '#aa0000', False: '#0000aa'}
        for particle in frame[1]:
            dots.append(ax.add_artist(Circle(particle[0], size*0.025, color=charge_colors[particle[1]>0])))
        return stream

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(-size,size)
    ax.set_ylim(-size,size)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_aspect('equal')
    color = 2 * np.log(np.hypot(Ex, Ey))
    stream = ax.streamplot(xv, yv, data[0][0][0], data[0][0][1], color=color, linewidth=1, cmap=cm.plasma,
                  density=2, arrowstyle='->', arrowsize=1.5)
    dots = []
    charge_colors = {True: '#aa0000', False: '#0000aa'}
    for particle in data[0][1]:
        dots.append(ax.add_artist(Circle(particle[0], size*0.025, color=charge_colors[particle[1]>0])))

    anim = animation.FuncAnimation(fig, animate, frames=data, interval=200, blit=False, repeat=True)
    anim.save('./animation.gif', fps=15, writer="pillow")
    np.save("sim.npy", np.array(data))
    print("done")


main()
