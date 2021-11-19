#run with current environment (-V) and in the current directory (-cwd)
#$ -V -cwd

#Request some time- min 15 mins - max 48 hours
#$ -l h_rt=48:00:00

#$ -l disk_out=output

#$ -l disk=40G

#$ -l disk_type=*

#Request some memory per core
#$ -l h_vmem=80G

#Get email at start and end of the job
#$ -m be


module load cuda

bash download_zenodo.sh download_skeletons.txt dataset 
