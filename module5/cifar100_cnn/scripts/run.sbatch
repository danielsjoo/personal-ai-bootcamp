#!/bin/bash
#SBATCH --job-name=train_cnn
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --ntasks=2
#SBATCH --gpus=rtx_5000_ada:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/train_sh_%j.log
#SBATCH --time=00:30:00
#SBATCH --mail-user=daniel.sh.joo@gmail.com
#SBATCH --mail-type=ALL

timestamp=$(date +'%m%d%H%M')
export NODE_HOSTNAME=$(hostname -s)

#run this from cifar100_cnn directory
mkdir -p logs

echo "Loading modules..."
module load CUDA/12.9.0
module load Python/3.12.3-GCCcore-13.3.0 

echo "Activating virtual environment..."
source ../venv/bin/activate

#tensorboard
tensorboard --logdir ./tb_logs &
TB_PID=$!
echo "to view tensorboard, run this command locally:"
echo "ssh -L 8080:${NODE_HOSTNAME}:6006 -J dsj22@bouchet.ycrc.yale.edu dsj22@${NODE_HOSTNAME}"

#nvidia-smi
nvitop -1 > logs/${timestamp}_gpu_log.csv &
NVSMI_PID=$!

#training script
srun --ntasks=2 --gpus-per-task=1 torchrun scripts/ddp_train.py ${timestamp} &
TRAIN_PID=$!

#pidstat
log_file="logs/${timestamp}_cpu_mem_log.csv"

# --- The Monitoring Function ---
# This function contains the core logic
monitor_process() {
    # Write the CSV header to the log file
    echo "timestamp,%cpu,%mem" > "$1"

    # Loop indefinitely to collect stats
    # The 'top' command provides the data.
    # 'grep' isolates the line for our specific PID.
    # 'awk' formats the line into a CSV row with a new timestamp.
    while true; do
        # Get the process stats from top
        stats=$(top -b -n 1 -p "$2" | grep "$2")

        # Check if the process is still running
        if [ -z "$stats" ]; then
            echo "Process $2 has finished. Stopping monit or."
            break
        fi

        # Extract CPU and Memory, then format with a new timestamp
        # NOTE: Column numbers in 'top' can vary slightly. 9 and 10 are common for %CPU and %MEM.
        # Verify these columns on your system if the output looks wrong.
        echo "$stats" | awk -v ts="$(date +%Y-%m-%dT%H:%M:%S)" '{print ts "," $9 "," $10}' >> "$1"

        # Wait for 5 seconds
        sleep 5
    done
}

# --- Execution ---
# Run the monitor function in the background
monitor_process "$log_file" "$TRAIN_PID" &
PIDSTAT_PID=$!

wait $TRAIN_PID

echo "Training finished. Cleaning up monitoring processes..."
kill $TB_PID
kill $NVSMI_PID
kill $PIDSTAT_PID

echo "Job complete."