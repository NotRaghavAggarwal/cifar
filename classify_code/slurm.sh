#!/bin/bash
#SBATCH --job-name=cifar_resnet_train          # Job name
#SBATCH --output=logs/cifar_resnet_train_%j.log  # Standard output and error log
#SBATCH --nodes=1
#SBATCH --ntasks=1                            # Number of tasks (1 task in this case)
#SBATCH --cpus-per-task=16                    # Number of CPUs per task (adjust as needed)
#SBATCH --mem=32G                           # Memory per node
#SBATCH --gres=gpu:1                         # Request 1 GPU
#SBATCH --partition=gpu                      # Specify the GPU partition, adjust if needed

#SBATCH --mail-type=ALL                       # Email notifications for job begin, end, and failure
#SBATCH --mail-user=aggarwalraghav29@gmail.com    # Replace with your email address

source /home/Student/s4655393/demo/bin/activate /home/Student/s4655393/cifar/classify_code

pyhon test_cifar_resnet.py 
