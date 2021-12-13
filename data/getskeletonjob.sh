#run with the current environment (-V) and in the current directory (-cwd)
#$ -V -cwd

#Request some time- min 15 mins - max 48 hours
#$ -l h_rt=48:00:00

#$ -l disk_out=output

#$ -l disk_type=*
#$ -l disk=10G
#Request some memory per core
#$ -l h_vmem=20G

#Get email at start and end of the job
#$ -m be


module load cuda

python -u  get_projectedAmplitudess.py
