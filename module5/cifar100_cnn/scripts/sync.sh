HPC_USER="dsj22"
HPC_ADDRESS="bouchet.ycrc.yale.edu"
PROJECT_DIR="/Users/danieljoo/Code/ai_bootcamp/module5/cifar100_cnn/scripts"
HPC_DIR="/home/dsj22/scratch_pi_br423/dsj22/cifar100"

rsync -avz --progress ${PROJECT_DIR} ${HPC_USER}@${HPC_ADDRESS}:${HPC_DIR}

# scp "${HPC_USER}@${HPC_ADDRESS}:${HPC_DIR}/hpc_sinfo.txt" \
#     "${PROJECT_DIR}/"