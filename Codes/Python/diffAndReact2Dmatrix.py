import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.stats
from matplotlib.animation import FuncAnimation

# Features ###################################################################
recordVideo = False # It slows down the script
recordTrajectories = True # It uses up memory

# Parameters #################################################################
num_steps = 1000 # Number of steps
D = 1.0  # Diffusion constant
meanEta = 0 # Spatial jump distribution paramenter
stdEta = 1 # Spatial jump distribution paramenter
meanCross = 0 # Crossing probability parameter
stdCross = 1 # Crossing probability parameter
num_particles = 1000 # Number of particles in the simulation
uby = 10 # Vertical Upper Boundary
lby = -10 # Vertical Lower Boundary
lbx = 0 # Horizontal Left Boundary
rbx = 50 # Horizontal Right Boundary
init_shift = 0 # It aggregates the initial positions of the particles around the centre of the domain
reflectedInward = 60 # Percentage of impacts from the fracture reflected again into the fracture
reflectedOutward = 70 # Percentage of impacts from the porous matrix reflected again into the porous matrix
animatedParticle = 3 # Index of the particle whose trajectory will be animated
fTstp = 10 # First time step to be recorded in the video
lTstp = 90 # Final time step to appear in the video

noise_strength = np.sqrt(2 * D)  # Strength of the noise term
crossOut = scipy.stats.norm.ppf(reflectedInward/100)
crossIn = scipy.stats.norm.ppf(reflectedOutward/100)
arrival = [False for _ in range(num_particles)]
endTstep = []

bouncesBackIn = 0
bouncesBackOut = 0
crossInToOut = 0
crossOutToIn = 0

# Case for recorded trajectories #############################################
if recordTrajectories:
    # Initialize arrays to store position data
    x = [np.zeros(num_steps) for _ in range(num_particles)]
    y0 = np.linspace(lby+init_shift, uby-init_shift, num_particles)
    y = [np.zeros(num_steps) for _ in range(num_particles)]
    for index, array in enumerate(y):
        array[0] = y0[index]

    # Simulate Langevin dynamics
    for n in range(num_particles):
        cross = False # After the particle crosses the fracture's walls once, it can freely move from fracture to matric and viceversa
        for i in range(1, num_steps):
            # Generate random forces (Gaussian white noise)
            eta_x = np.random.normal(meanEta, stdEta)
            eta_y = np.random.normal(meanEta, stdEta)
            
            x[n][i] = x[n][i-1] + noise_strength*eta_x
            y[n][i] = y[n][i-1] + noise_strength*eta_y

            if x[n][i] < lbx:
                x[n][i] = x[n][i] + 2*(lbx-x[n][i])
            # The following condition compares the particle time step i with a sample vector which stores the reflecting probabilities randomly arranged
            # The particle gets reflected until it crosses the fracture's wall. Once it crossed it moves freely between fractures' boundaries
            crossProb = np.random.normal(meanCross, stdCross)
            if cross == False:
                # The particle bounces back in against the fracture's wall
                if y[n][i] > uby and crossProb < crossOut:
                   y[n][i] = y[n][i] - 2*(y[n][i]-uby)
                   bouncesBackIn = bouncesBackIn+1
                if y[n][i] < lby and crossProb < crossOut:
                   y[n][i] = y[n][i] + 2*(lby-y[n][i])
                   bouncesBackIn = bouncesBackIn+1
                # The particle leaves the fracture
                if (y[n][i] > uby or y[n][i] < lby) and crossProb > crossOut:
                    cross = True
                    crossInToOut = crossInToOut+1
            if cross == True:
                # The particle bounces against the fracture's wall
                if y[n][i] <= uby and crossProb < crossIn:
                   y[n][i] =  y[n][i] + 2*(uby-y[n][i])
                   bouncesBackOut = bouncesBackOut+1
                if y[n][i] >= lby and crossProb < crossIn:
                   y[n][i] =  y[n][i] - 2*(y[n][i]-lby)
                   bouncesBackOut = bouncesBackOut+1
                # The particle enters the fracture
                if (y[n][i] <= uby or y[n][i] >= lby) and crossProb > crossIn:
                    cross = False
                    crossOutToIn = crossOutToIn+1
            if x[n][i] > rbx:
                x[n] = x[n][:i]
                y[n] = y[n][:i]
                if y[n][i-1] < uby or y[n][i-1] > lby:
                    arrival[n] = True
                break

    bc_time = [len(value)/num_steps for index, value in enumerate(x) if (len(value)<num_steps and arrival[index]== True)]
    bc_time.sort()
    cum_part = [i/num_particles for i in range(len(bc_time))]

    # Plot the trajectory
    plt.figure(figsize=(8, 8))
    for i in range(num_particles):
        plt.plot(x[i], y[i], lw=0.5)
    plt.axhline(y=uby, color='r', linestyle='--', linewidth=2)
    plt.axhline(y=lby, color='r', linestyle='--', linewidth=2)
    plt.title("2D Diffusion Process (Langevin Equation)")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.grid(True)
    plt.show()

    if recordVideo:
        # Animate the trajectory
        # Set up the figure, axis, and plot element to animate
        fig, ax = plt.subplots()
        line, = ax.plot([], [], lw=2)
        # Set the plot limits
        ax.set_xlim(lbx, rbx)
        ax.set_ylim(lby, uby)
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

    # Plot Breakthrough curve
    plt.figure(figsize=(8, 8))
    plt.plot(bc_time, cum_part, lw=0.5)
    plt.title("Breakthorugh curve")
    plt.xlabel("Time step")
    plt.ylabel("CDF")
    plt.grid(True)
    plt.show()

# Case without recorded trajectories #########################################
else:
    # Initialize arrays to store position data
    x0 = np.zeros(num_particles)
    y0 = np.linspace(lby+init_shift, uby-init_shift, num_particles)

    # Simulate Langevin dynamics
    for n in range(num_particles):
        x = x0[n]
        y = y0[n]
        cross = False # After the particle crosses the fracture's walls once, it can freely move from fracture to matric and viceversa
        for i in range(1, num_steps):
            # Generate random forces (Gaussian white noise)
            eta_x = np.random.normal(meanEta, stdEta)
            eta_y = np.random.normal(meanEta, stdEta)
            
            x = x + noise_strength*eta_x
            y = y + noise_strength*eta_y

            if x < lbx:
                x = x - noise_strength*eta_x
            # The following condition compares the particle time step i with a sample vector which stores the reflecting probabilities randomly arranged
            # The particle gets reflected until it crosses the fracture's wall. Once it crossed it moves freely between fractures' boundaries
            crossProb = np.random.normal(meanCross, stdCross)
            if cross == False:
                # The particle bounces back in against the fracture's wall
                if (y > uby or y < lby) and crossProb < crossOut:
                    y = y - noise_strength*eta_y
                    bouncesBackIn = bouncesBackIn+1
                # The particle leaves the fracture
                if (y > uby or y < lby) and crossProb > crossOut:
                    cross = True
                    crossInToOut = crossInToOut+1
            if cross == True:
                # The particle bounces against the fracture's wall
                if (y < uby or y > lby) and crossProb < crossIn:
                    y = y - noise_strength*eta_y
                    bouncesBackOut = bouncesBackOut+1
                # The particle enters the fracture
                if (y < uby or y > lby) and crossProb > crossIn:
                    cross = False
                    crossOutToIn = crossOutToIn+1
            if x > rbx:
                if y < uby or y > lby:
                    arrival[n] = True
                    endTstep.extend([i])
                break

    endTstep.sort()
    endTstep = [i/num_steps for i in endTstep]
    cum_part = [i/num_particles for i in range(len(endTstep))]

    with open("BreakthroughCurve.txt", "w") as file:
        for time, prob in zip(endTstep, cum_part):
            file.write(f"{time}\t{prob}\n")

    # Plot Breakthrough curve
    plt.figure(figsize=(8, 8))
    plt.plot(endTstep, cum_part, lw=0.5)
    plt.title("Breakthorugh curve")
    plt.xlabel("Time step")
    plt.ylabel("CDF")
    plt.grid(True)
    plt.show()

# Statistics to double check the results of the simulations against the inputs
print("Effective bounce-in fraction: ", 100*bouncesBackIn/(bouncesBackIn+crossInToOut))
print("Effective bounce-out fraction: ", 100*bouncesBackOut/(bouncesBackOut+crossOutToIn))
print("Time scale: L^2/D", (rbx-lbx)**2/D**2)