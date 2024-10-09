import numpy as np
import math
import scipy.stats
import time

# Features ###################################################################
plotCharts = True # It controls graphical features (disable when run on HPC)
recordVideo = False # It slows down the script
recordTrajectories = False # It uses up memory

if plotCharts:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

# Parameters #################################################################
num_steps = 10000 # Number of steps
Dm = 0.1  # Diffusion for particles moving in the porous matrix
Df = 0.1  # Diffusion for particles moving in the fracture
dt = 1 # Time step
meanEta = 0 # Spatial jump distribution paramenter
stdEta = 1 # Spatial jump distribution paramenter
meanCross = 0 # Crossing probability parameter
stdCross = 1 # Crossing probability parameter
num_particles = 10000 # Number of particles in the simulation
uby = 1 # Vertical Upper Boundary
lby = -1 # Vertical Lower Boundary
lbx = 0 # Horizontal Left Boundary
rbx = 10 # Horizontal Right Boundary
init_shift = 0 # It aggregates the initial positions of the particles around the centre of the domain
reflectedInward = 90 # Percentage of impacts from the fracture reflected again into the fracture
reflectedOutward = 30 # Percentage of impacts from the porous matrix reflected again into the porous matrix
animatedParticle = 0 # Index of the particle whose trajectory will be animated
fTstp = 0 # First time step to be recorded in the video
lTstp = 90 # Final time step to appear in the video

noiseMatrix = np.sqrt(2*Dm*dt)  # Strength of the noise term for particle in the porous matrix
noiseFracture = np.sqrt(2*Df*dt)  # Strength of the noise term for particle in the fracture
crossOut = scipy.stats.norm.ppf(reflectedInward/100)
crossIn = scipy.stats.norm.ppf(reflectedOutward/100)
inFraRbx = [False for _ in range(num_particles)]
bc_time = []

bouncesBackIn = 0
bouncesBackOut = 0
crossInToOut = 0
crossOutToIn = 0
staysIn = 0
staysOut = 0
outsideFractureUp = False
outsideFractureDown = False

start_time = time.time()
# If recording trajectories is enabled #############################################
if recordTrajectories:
    # Initialize arrays to store position data
    x = [np.zeros(num_steps) for _ in range(num_particles)]
    y0 = np.linspace(lby+init_shift, uby-init_shift, num_particles)
    y = [np.zeros(num_steps) for _ in range(num_particles)]
    for index, array in enumerate(y):
        array[0] = y0[index]

    # Simulate Langevin dynamics
    for n in range(num_particles):
        outsideFractureUp = False # After the particle crosses the fracture's walls once, it can freely move from fracture to matric and viceversa
        outsideFractureDown = False # After the particle crosses the fracture's walls once, it can freely move from fracture to matric and viceversa
        noise = noiseFracture
        for i in range(1, num_steps):
            # Generate random forces (Gaussian white noise)
            eta_x = np.random.normal(meanEta, stdEta)
            eta_y = np.random.normal(meanEta, stdEta)
            
            x[n][i] = x[n][i-1] + noise*eta_x
            y[n][i] = y[n][i-1] + noise*eta_y

            if x[n][i] < lbx:
               x[n][i] = x[n][i] + 2*(lbx-x[n][i])
            # The following condition compares the particle time step i with a sample vector which stores the reflecting probabilities randomly arranged
            # The particle gets reflected until it crosses the fracture's wall. Once it crossed it moves freely between fractures' boundaries
            crossProb = np.random.normal(meanCross, stdCross)
            if (outsideFractureUp == False and outsideFractureDown == False):
                # The particle leaves the fracture
                if y[n][i] > uby and crossProb > crossOut:
                   outsideFractureUp = True
                   crossInToOut = crossInToOut+1
                   noise = noiseMatrix
                elif y[n][i] < lby and crossProb > crossOut:
                   outsideFractureDown = True
                   crossInToOut = crossInToOut+1
                   noise = noiseMatrix
                # The particle bounces back in against the fracture's wall
                elif y[n][i] > uby and crossProb < crossOut:
                   y[n][i] = y[n][i] - 2*(y[n][i]-uby)
                   bouncesBackIn = bouncesBackIn+1
                elif y[n][i] < lby and crossProb < crossOut:
                   y[n][i] = y[n][i] + 2*(lby-y[n][i])
                   bouncesBackIn = bouncesBackIn+1
                # The particle remains in the fracture without bouncing
                else:
                   staysIn = staysIn+1
            else:
                # The particle enters the fracture
                if (y[n][i] <= uby and y[n][i] >= lby) and crossProb > crossIn:
                   outsideFractureUp = False
                   outsideFractureDown = False
                   crossOutToIn = crossOutToIn+1
                   noise = noiseFracture
                # The particle bounces against the fracture's wall
                elif (outsideFractureUp == True and y[n][i] <= uby) and crossProb < crossIn:
                   y[n][i] =  y[n][i] + 2*(uby-y[n][i])
                   bouncesBackOut = bouncesBackOut+1
                elif (outsideFractureDown == True and y[n][i] >= lby) and crossProb < crossIn:
                   y[n][i] =  y[n][i] - 2*(y[n][i]-lby)
                   bouncesBackOut = bouncesBackOut+1
                # The particle remains in the porous matrix without bouncing
                else:
                   staysOut = staysOut+1
            if (x[n][i] > rbx and y[n][i] < uby and y[n][i] > lby):
               inFraRbx[n] = True
               x[n] = x[n][:i]
               y[n] = y[n][:i]
               break
            elif x[n][i] > rbx:
               x[n] = x[n][:i]
               y[n] = y[n][:i]               
               break

    end_time = time.time()
    execution_time = end_time - start_time

    bc_time_inFra = [len(value)/num_steps for index, value in enumerate(x) if (len(value)<num_steps and inFraRbx[index]==True)]
    bc_time_inFra.sort()
    cum_part_inFra = [i/num_particles for i in range(len(bc_time_inFra))]

    bc_time = [x[i][-1]/num_steps for i in range(num_particles) if (len(x[i])<num_steps)]
    bc_time.sort()
    cum_part = [i/num_particles for i in range(len(bc_time))]

# If recording trajectories is NOT enabled #########################################
else:
    # Initialize arrays to store position data
    x0 = np.zeros(num_particles)
    y0 = np.linspace(lby+init_shift, uby-init_shift, num_particles)

    # Simulate Langevin dynamics
    for n in range(num_particles):
        x = x0[n]
        y = y0[n]
        outsideFractureUp = False # After the particle crosses the fracture's walls once, it can freely move from fracture to matric and viceversa
        outsideFractureDown = False # After the particle crosses the fracture's walls once, it can freely move from fracture to matric and viceversa
        for i in range(1, num_steps):
            # Generate random forces (Gaussian white noise)
            eta_x = np.random.normal(meanEta, stdEta)
            eta_y = np.random.normal(meanEta, stdEta)
            
            x = x + noiseMatrix*eta_x
            y = y + noiseMatrix*eta_y

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
               inFraRbx[n] = True
               bc_time.extend([i])
               break
            elif x > rbx:
               bc_time.extend([i])
               break

    end_time = time.time()
    execution_time = end_time - start_time

    bc_time_inFra = [value/num_steps for index, value in enumerate(bc_time) if inFraRbx[index]==True]
    bc_time_inFra.sort()
    cum_part_inFra = [i/num_particles for i in range(len(bc_time_inFra))]

    bc_time = [i/num_steps for i in bc_time]
    bc_time.sort()
    cum_part = [i/num_particles for i in range(len(bc_time))]

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

if plotCharts:
    # Plot Breakthrough curve
    plt.figure(figsize=(8, 8))
    plt.plot(bc_time, cum_part, lw=0.5)
    plt.title("CDF")
    plt.xlabel("Time step")
    plt.ylabel("CDF")
    plt.grid(True)
    plt.show()

    # Plot Breakthrough curve
    plt.figure(figsize=(8, 8))
    plt.plot(bc_time, [1-i for _, i in enumerate(cum_part)], lw=0.5)
    plt.title("1-CDF")
    plt.xlabel("Time step")
    plt.ylabel("CDF")
    plt.grid(True)
    plt.show()

    # Plot Breakthrough curve for particles that reached the right boundary from within the fracture
    plt.figure(figsize=(8, 8))
    plt.plot(bc_time_inFra, cum_part_inFra, lw=0.5)
    plt.title("BC for particles that reached the rx boundary from within the fracture")
    plt.xlabel("Time step")
    plt.ylabel("CDF")
    plt.grid(True)
    plt.show()

# Statistics to double check the results of the simulations against the inputs
print("Average # of steps: ", (num_particles+staysIn+staysOut+bouncesBackIn+bouncesBackOut+crossInToOut+crossOutToIn)/num_particles)
print("Effective bounce-in fraction: ", 100*bouncesBackIn/(bouncesBackIn+crossInToOut))
print("Effective bounce-out fraction: ", 100*bouncesBackOut/(bouncesBackOut+crossOutToIn)) if bouncesBackOut+crossOutToIn>0 else print("The particles never leave the fracture")
print("Horizontal time scale for particles within the fracture: L^2/Df", (rbx-lbx)**2/Df**2)
print("Verticlal time scale for particles within the fracture: L^2/Df", (uby-lby)**2/Df**2)
print(f"Execution time: {execution_time:.6f} seconds")