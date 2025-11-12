#!/bin/bash

# Usage: ./submit_slurm_job.sh path/to/script.py email_id num_hours

set -e

# Load .env in the current calling directory
if [ -f ".env" ]; then
    export $(grep -v '^#' .env | xargs)
else
    echo ".env file not found in current directory."
    exit 1
fi

if [ "$#" -lt 3 ]; then
    echo "Usage: $0 <python_file> <email id> <num_hours> [python_args...]"
    exit 1
fi

PYTHON_FILE="$1"
EMAIL_ID="$2"
NUM_HOURS="$3"
shift 3
PYTHON_ARGS="$@"

BASENAME=$(basename "$PYTHON_FILE" .py)
JOB_NAME="$BASENAME"
TIME_FORMAT="0-${NUM_HOURS}:00:00"

SLURM_SCRIPT="temp_slurm_${JOB_NAME}.sh"

cleanup() {
    if [ -f "$SLURM_SCRIPT" ]; then
        rm "$SLURM_SCRIPT"
    fi
}
trap cleanup EXIT


cat <<EOL > "$SLURM_SCRIPT"
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=${NUM_CORES}
#SBATCH --cluster=smp
#SBATCH --partition=smp
#SBATCH --mail-user=${EMAIL_ID}@pitt.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --time=${TIME_FORMAT}
#SBATCH --mem=16384
#SBATCH --output=${LOG_DIR}/%j.out
#SBATCH --error=${LOG_DIR}/%j.err

cd .venv/bin
source activate
cd ../..
python3 ${PYTHON_FILE} ${PYTHON_ARGS}
EOL

echo "======== SLURM script generated: ========"
cat "$SLURM_SCRIPT"
echo "========================================="

SBATCH_OUTPUT=$(sbatch "$SLURM_SCRIPT" || true)

if [ -z "$SBATCH_OUTPUT" ]; then
    echo "Error: sbatch failed to submit the job."
    exit 1
fi

JOB_ID=$(echo "$SBATCH_OUTPUT" | awk '{print $4}')

if [ -z "$JOB_ID" ]; then
    echo "Error: Failed to extract Job ID from sbatch output."
    exit 1
fi

echo "Job submitted with ID ${JOB_ID}. The job should be running now."

echo "Current job status:"
scontrol -M smp show job "$JOB_ID"