#PBS -l mem=16gb
#PBS -l nodes=1:ppn=2 
#PBS -l walltime=140:00:00
#PBS -m abe
#PBS -e /projects/ibm_aihl/whogan/acronym_resolution/ar_progress.errors
#PBS -o /projects/ibm_aihl/whogan/acronym_resolution/ar_progress.outputs
#PBS -M whogan@ucsd.edu
#PBS -N gpt2_ar_v1

# Manual env setup
source /opt/miniconda3/bin/activate latest_python

# Processing outputs
pwd > /projects/ibm_aihl/whogan/acronym_resolution/ar_progress.txt
echo started ... >> /projects/ibm_aihl/whogan/acronym_resolution/ar_progress.txt
hostname >> /projects/ibm_aihl/whogan/acronym_resolution/ar_progress.txt

# Main file
python /projects/ibm_aihl/whogan/acronym_resolution/gpt2_fine_tune.py

# Final processing output
echo done ... >> /projects/ibm_aihl/whogan/acronym_resolution/ar_progress.txt
