import numpy as np
import math
import scipy.stats
from concurrent.futures import ProcessPoolExecutor

# Features ###################################################################
localRun = False # Disable graphical features for running on HPC
recordVideo = False # It slows down the script
recordTrajectories = False # It uses up memory

if localRun:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

# Parameters #################################################################
num_steps = 1000 # Number of steps
D = 0.1  # Diffusion constant
meanEta = 0 # Spatial jump distribution paramenter
stdEta = 1 # Spatial jump distribution paramenter
meanCross = 0 # Crossing probability parameter
stdCross = 1 # Crossing probability parameter
num_particles = 100 # Number of particles in the simulation
uby = 1 # Vertical Upper Boundary
lby = -1 # Vertical Lower Boundary
lbx = 0 # Horizontal Left Boundary
rbx = 20 # Horizontal Right Boundary
init_shift = 0.5 # It aggregates the initial positions of the particles around the centre of the domain
reflectedInward = 90 # Percentage of impacts from the fracture reflected again into the fracture
reflectedOutward = 30 # Percentage of impacts from the porous matrix reflected again into the porous matrix
animatedParticle = 0 # Index of the particle whose trajectory will be animated
fTstp = 0 # First time step to be recorded in the video
lTstp = 90 # Final time step to appear in the video

noise_strength = np.sqrt(2 * D)  # Strength of the noise term
crossOut = scipy.stats.norm.ppf(reflectedInward/100)
crossIn = scipy.stats.norm.ppf(reflectedOutward/100)
arrival = [False for _ in range(num_particles)]
bc_time = []
brktrc = []

bouncesBackIn = 0
bouncesBackOut = 0
crossInToOut = 0
crossOutToIn = 0
staysIn = 0
staysOut = 0
outsideFractureUp = False
outsideFractureDown = False

# Initialize arrays to store position data
x0 = np.zeros(num_particles)
y0 = np.linspace(lby+init_shift, uby-init_shift, num_particles)

def oneStep(x0, y0):

   out = False
   outsideFractureUp = False # After the particle crosses the fracture's walls once, it can freely move from fracture to matric and viceversa
   outsideFractureDown = False # After the particle crosses the fracture's walls once, it can freely move from fracture to matric and viceversa

   # Generate random forces (Gaussian white noise)
   eta_x = np.random.normal(meanEta, stdEta)
   eta_y = np.random.normal(meanEta, stdEta)
   
   x = x0 + noise_strength*eta_x
   y = y0 + noise_strength*eta_y

   if x < lbx:
      x = x + 2*(lbx-x)
   # The following condition compares the particle time step i with a sample vector which stores the reflecting probabilities randomly arranged
   # The particle gets reflected until it crosses the fracture's wall. Once it crossed it moves freely between fractures' boundaries
   crossProb = np.random.normal(meanCross, stdCross)
   if (outsideFractureUp == False and outsideFractureDown == False):
      # The particle leaves the fracture
      if y > uby and crossProb > crossOut:
         outsideFractureUp = True
         crossInToOut = crossInToOut+1
      elif y < lby and crossProb > crossOut:
         outsideFractureDown = True
         crossInToOut = crossInToOut+1
      # The particle bounces back in against the fracture's wall
      elif y > uby and crossProb < crossOut:
         y = y - 2*(y-uby)
         bouncesBackIn = bouncesBackIn+1
      elif y < lby and crossProb < crossOut:
         y = y + 2*(lby-y)
         bouncesBackIn = bouncesBackIn+1
      # The particle remains in the fracture without bouncing
      else:
         staysIn = staysIn+1
   else:
      # The particle enters the fracture
      if (y <= uby and y >= lby) and crossProb > crossIn:
         outsideFractureUp = False
         outsideFractureDown = False
         crossOutToIn = crossOutToIn+1
      # The particle bounces against the fracture's wall
      elif (outsideFractureUp == True and y <= uby) and crossProb < crossIn:
         y =  y + 2*(uby-y)
         bouncesBackOut = bouncesBackOut+1
      elif (outsideFractureDown == True and y >= lby) and crossProb < crossIn:
         y =  y - 2*(y-lby)
         bouncesBackOut = bouncesBackOut+1
      # The particle remains in the porous matrix without bouncing
      else:
         staysOut = staysOut+1
   if (x > rbx and y < uby and y > lby):
      out = True
      return out

def oneTrajectory(x0, y0):
   out = False
   while out == False and i <= num_steps:
      oneStep(x0, y0)
   return i

def oneCore(max_workers=4):
    """Parallelize a loop over many steps."""
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all steps to be processed in parallel
        brktrc = list(executor.map(oneTrajectory, x0, y0))
        brktrc.sort()
        brktrc = [i/num_steps for i in brktrc]
        cum_part = [i/num_particles for i in range(len(brktrc))]
    return brktrc, cum_part

bc_time, cum_part = oneCore(max_workers=4)

# Plot section ###############################################################

# Print BC on external txt file
with open("BreakthroughCurve.txt", "w") as file:
    for time, prob in zip(bc_time, cum_part):
        file.write(f"{time}\t{prob}\n")

# Plot the trajectory
if recordTrajectories:
    plt.figure(figsize=(8, 8))
    for i in range(num_particles):
        plt.plot(x[i], y[i], lw=0.5)
    plt.axhline(y=uby, color='r', linestyle='--', linewidth=2)
    plt.axhline(y=lby, color='r', linestyle='--', linewidth=2)
    plt.axvline(x=lbx, color='black', linestyle='-', linewidth=2)
    plt.title("2D Diffusion Process (Langevin Equation)")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.grid(True)
    plt.show()

# Record video
if recordVideo:
    # Animate the trajectory
    # Set up the figure, axis, and plot element to animate
    fig, ax = plt.subplots()
    line, = ax.plot([], [], lw=2)
    # Set the plot limits
    ax.set_xlim(min(x[animatedParticle]), max(x[animatedParticle]))
    ax.set_ylim(min(y[animatedParticle]), max(y[animatedParticle]))
    plt.axhline(y=uby, color='r', linestyle='--', linewidth=2)
    plt.axhline(y=lby, color='r', linestyle='--', linewidth=2)
    plt.axvline(x=lbx, color='black', linestyle='-', linewidth=2)
    plt.grid(True)
    # Initialization function: plot the background of each frame
    def init():
        line.set_data([], [])
        return line,
    # Animation function: this is called sequentially
    def animate(i):
        line.set_data(x[animatedParticle][fTstp:i], y[animatedParticle][fTstp:i])  # Update the data to show part of the line
        return line,
    # Call the animator
    ani = FuncAnimation(fig, animate, init_func=init, frames=len(x[animatedParticle][fTstp:lTstp]), interval=400, blit=True)
    ani.save('animated_chart.mp4', writer='ffmpeg', fps=20)
    plt.show()

if localRun:
    # Plot Breakthrough curve
    plt.figure(figsize=(8, 8))
    plt.plot(bc_time, cum_part, lw=0.5)
    plt.title("Breakthorugh curve")
    plt.xlabel("Time step")
    plt.ylabel("CDF")
    plt.grid(True)
    plt.show()

# Statistics to double check the results of the simulations against the inputs
print("Effective steps (only if particles do not get absorbed by right boundary): ", (num_particles+staysIn+staysOut+bouncesBackIn+bouncesBackOut+crossInToOut+crossOutToIn)/num_particles)
print("Effective bounce-in fraction: ", 100*bouncesBackIn/(bouncesBackIn+crossInToOut))
print("Effective bounce-out fraction: ", 100*bouncesBackOut/(bouncesBackOut+crossOutToIn))
print("Time scale: L^2/D", (rbx-lbx)**2/D**2)
