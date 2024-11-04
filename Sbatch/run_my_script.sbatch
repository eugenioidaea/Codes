#!/bin/bash
#SBATCH --job-name = eugenioPythonJob				# Job name
#SBATCH --output = output.log					# Standard output and error log
#SBATCH -p generic                          			# Queue
#SBATCH -N 1							# Number of nodes
#SBATCH -n 1							# Number of cores
#SBATCH -- ntasks =1						# Number of tasks
#SBATCH -- cpus - per - task =1					# CPUs per task
#SBATCH --mem - per - cpu = 1000000				# megabytes
#SBATCH -t 10-00:00						# Duration D-HH:MM
#SBATCH --job - name = testP8					# Job name
#SBATCH -- mail - type = END , FAIL				# Notification
#SBATCH -- mail - user = eugenio.pescimoro@idaea.csic.es	# email address

# Commands to run the job
#module load gnu8/8.3.0
#module load Python/3.8.6-GCCcore-10.2.0
#module load OpenFOAM

module load OpenFOAM/8-foss-2020a

. $FOAM_BASH

# Run the Python script
python3 ~/Git/Hub/IDAEA/Codes/Python/diffAndReact2Dmatrix.py
