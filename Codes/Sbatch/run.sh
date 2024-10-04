# !/ bin / bash
#
# SBATCH -p generic                          	# Queue
# SBATCH -N 1					# Number of nodes
# SBATCH -n 8					# Number of cores
# SBATCH -- ntasks =1				# Number of tasks
# SBATCH -- cpus - per - task =1		# CPUs per task
# SBATCH --mem - per - cpu =2625		# megabytes
# SBATCH -t 10 -00:00				# Duration D-HH:MM
# SBATCH --job - name = testP8			# Job name
# SBATCH -- mail - type = END , FAIL		# Notification
# SBATCH -- mail - user = agnes@mail . com	# email address
# SBATCH -- comment = openFOAM

# Commands to run the job .
module load OpenFOAM
. $FOAM_BASH
mpirun -- oversubscribe - np 8 simpleFoam - parallel
# End of the script .
