HPC_USER="dsj22"
HPC_ADDRESS="bouchet.ycrc.yale.edu"
PROJECT_DIR="/Users/danieljoo/Code/ai_bootcamp/module5/pytorch_gpt/train_gpt.py"
HPC_DIR="/home/dsj22/scratch_pi_br423/dsj22/gpt"

rsync -avz --progress ${PROJECT_DIR} ${HPC_USER}@${HPC_ADDRESS}:${HPC_DIR}

# scp "${HPC_USER}@${HPC_ADDRESS}:${HPC_DIR}/hpc_sinfo.txt" \
#     "${PROJECT_DIR}/"