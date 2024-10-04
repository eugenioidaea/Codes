# !/ bin / bash
# SBATCH --job-name = eugenioPythonJob				# Job name
# SBATCH --output = output.log					# Standard output and error log
# SBATCH -p generic                          			# Queue
# SBATCH -N 1							# Number of nodes
# SBATCH -n 8							# Number of cores
# SBATCH -- ntasks =1						# Number of tasks
# SBATCH -- cpus - per - task =1				# CPUs per task
# SBATCH --mem - per - cpu =2625				# megabytes
# SBATCH -t 0-24:00						# Duration D-HH:MM
# SBATCH --job - name = testP8					# Job name
# SBATCH -- mail - type = END , FAIL				# Notification
# SBATCH -- mail - user = eugenio.pescimoro@idaea.csic.es	# email address

# Commands to run the job
module load python3/3.12

# Run the Python script
python3 ~/Git/Hub/IDAEA/Codes/Python/diffAndReact2Dmatrix.py
