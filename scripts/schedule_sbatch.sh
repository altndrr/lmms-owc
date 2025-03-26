#!/usr/bin/env bash
set -o errexit
set -o nounset
set -o pipefail

if [[ "${TRACE-0}" == "1" ]]; then
    set -o xtrace
fi

if [[ "${1-}" =~ ^-*h(elp)?$ ]]; then
    echo 'usage: schedule_sbatch.sh [-h] [--partition PARTITION] [--account ACCOUNT]
                                [--nodes NODES] [--cpu CPU] [--gpu GPU] [--mem MEM]
                                [--time TIME] [--name NAME] [--models MODELS]
                                [--tasks TASKS] [--limit LIMIT] [--model-args ARGS]
                                [--no-samples] [--no-wandb] [--output OUTPUT]

Schedule a (slurm) batch of runs with the defined models and tasks.

Slurm options:
    -p, --partition <PARTITION>  Partition to use
    -A, --account <ACCOUNT>      Account to use
    -c, --cpu <CPU>              Number of CPUs per task to use (default: 12)
    -g, --gpu <GPU>              Number of GPUs to use (default: a100.40:8)
    -m, --mem <MEM>              Memory limit (default: 128G)
    -t, --time <TIME>            Time limit (default: 02:00:00)
    -n, --name <NAME>            Name of the job (default: schedule)

Evaluation options:
    --models <MODELS>            List of comma-separated models to run
    --tasks <TASKS>              List of comma-separated tasks to run
    --limit <LIMIT>              Limit the number of samples per task
    --models-args <ARGS>         Comma-separated extra args for the models
    --no-samples                 Disable logging samples to disk
    --no-wandb                   Disable logging to Weights & Biases
    -o --output <OUTPUT>         Results output dir (default: "logs/schedule")

'
    exit
fi

cd "$(dirname "$0")"
while [ "$(find . -maxdepth 1 -name pyproject.toml | wc -l)" -ne 1 ]; do cd ..; done

# Set default values
SLURM_NODES='1'
SLURM_TASKS='1'
SLURM_CPU='12'
SLURM_GPU='a100.40:8'
SLURM_MEM='128G'
SLURM_TIME='02:00:00'
SLURM_NAME='schedule'
SLURM_OUTPUT='./logs/slurm/%A_%a.out'
SLURM_ERROR='./logs/slurm/%A_%a.err'

EVAL_MODELS=""
EVAL_MODELS_ARGS=""
EVAL_OUTPUT_DIR=logs/schedule
EVAL_TASKS=""
EVAL_SAMPLES_LIMIT=""
EVAL_SAMPLES_LOGGING=true
EVAL_WANDB_LOGGING=true
EVAL_WANDB_ARGS="project=lmms-owc,job_type=eval"

main() {
    while [[ $# -gt 0 ]]; do
    # Parse arguments
        if [[ $1 == "--" ]]; then
            shift
            break
        fi
        case $1 in
            -p|--partition) SLURM_PARTITION="$2"; shift 2 ;;
            -A|--account) SLURM_ACCOUNT="$2"; shift 2 ;;
            -c|--cpu) SLURM_CPU="$2"; shift 2 ;;
            -g|--gpu) SLURM_GPU="$2"; shift 2 ;;
            -m|--mem) SLURM_MEM="$2"; shift 2 ;;
            -t|--time) SLURM_TIME="$2"; shift 2 ;;
            -n|--name) SLURM_NAME="$2"; shift 2 ;;
            --models) EVAL_MODELS="$2"; shift 2 ;;
            --tasks) EVAL_TASKS="$2"; shift 2;;
            --limit) EVAL_SAMPLES_LIMIT="$2"; shift 2;;
            --model-args|--models-args) EVAL_MODELS_ARGS="$2"; shift 2;;
            --no-samples) EVAL_SAMPLES_LOGGING=false; shift;;
            --no-wandb) EVAL_WANDB_LOGGING=false; shift;;
            -o|--output) EVAL_OUTPUT_DIR="$2"; shift 2 ;;
            *) echo "Error: unknown option: $1" >&2; exit 1 ;;
        esac
    done

    # Split comma-separated values into array
    IFS=',' read -ra EVAL_MODELS_ARRAY <<< "$EVAL_MODELS"
    IFS=',' read -ra EVAL_TASKS_ARRAY <<< "$EVAL_TASKS"

    # Count total number of jobs
    NUM_JOBS=$((${#EVAL_MODELS_ARRAY[@]} * ${#EVAL_TASKS_ARRAY[@]}))

    if [[ $NUM_JOBS -eq 0 ]]; then
        echo "Error: no jobs found to run. Please specify models and tasks to run." >&2
        exit 1
    fi

    # Run command
    sbatch <<EOT
#!/bin/bash
#SBATCH --partition=$SLURM_PARTITION
#SBATCH --account=$SLURM_ACCOUNT
#SBATCH --nodes=$SLURM_NODES
#SBATCH --array=1-$NUM_JOBS
#SBATCH --ntasks-per-node=$SLURM_TASKS
#SBATCH --cpus-per-task=$SLURM_CPU
#SBATCH --gres=gpu:$SLURM_GPU
#SBATCH --mem=$SLURM_MEM
#SBATCH --signal=B:SIGTERM@300
#SBATCH --time=$SLURM_TIME
#SBATCH --job-name=$SLURM_NAME
#SBATCH --output=$SLURM_OUTPUT
#SBATCH --error=$SLURM_ERROR

# Pass env variables
ACCELERATE_MAIN_PROCESS_PORT=\$((RANDOM % (50000 - 30000 + 1) + 30000))
ACCELERATE_NUM_PROCESSES=\$(nvidia-smi --list-gpus | wc -l)
EVAL_MODELS="$EVAL_MODELS"
EVAL_MODELS_ARGS="$EVAL_MODELS_ARGS"
EVAL_OUTPUT_DIR=$EVAL_OUTPUT_DIR
EVAL_TASKS="$EVAL_TASKS"
EVAL_SAMPLES_LIMIT=$EVAL_SAMPLES_LIMIT
EVAL_SAMPLES_LOGGING=$EVAL_SAMPLES_LOGGING
EVAL_WANDB_LOGGING=$EVAL_WANDB_LOGGING
EVAL_WANDB_ARGS="$EVAL_WANDB_ARGS"

# Split comma-separated values into array
IFS=',' read -ra EVAL_MODELS_ARRAY <<< "\$EVAL_MODELS"
IFS=',' read -ra EVAL_TASKS_ARRAY <<< "\$EVAL_TASKS"

if [ "\$EVAL_SAMPLES_LOGGING" = false ]; then
    echo "[warn] Skipping samples logging!"
fi

if [ "\$EVAL_WANDB_LOGGING" = false ]; then
    echo "[warn] Skipping wandb logging!"
fi

# Calculate the indices based on SLURM_ARRAY_TASK_ID
TASK_INDEX=\$(( (\$SLURM_ARRAY_TASK_ID - 1) / \${#EVAL_MODELS_ARRAY[@]} ))
MODEL_INDEX=\$(( (\$SLURM_ARRAY_TASK_ID - 1) % \${#EVAL_MODELS_ARRAY[@]} ))

# Get the specific task and model for this array job
task="\${EVAL_TASKS_ARRAY[TASK_INDEX]}"
model="\${EVAL_MODELS_ARRAY[MODEL_INDEX]}"

# Error checking
if [ -z "\$task" ] || [ -z "\$model" ]; then
    echo "Error: Invalid SLURM_ARRAY_TASK_ID (\$SLURM_ARRAY_TASK_ID) resulted in empty task (\$task) or model (\$model)"
    exit 1
fi

EVAL_EXTRA_ARGS=""
EVAL_EXTRA_ARGS="\$EVAL_EXTRA_ARGS --batch_size 1"
EVAL_EXTRA_ARGS="\$EVAL_EXTRA_ARGS --output_path \$EVAL_OUTPUT_DIR/\$task/\$model"

if [ "\$EVAL_MODELS_ARGS" ]; then
    EVAL_EXTRA_ARGS="\$EVAL_EXTRA_ARGS --model_args \$EVAL_MODELS_ARGS"
fi

if [ "\$EVAL_SAMPLES_LIMIT" ]; then
    EVAL_EXTRA_ARGS="\$EVAL_EXTRA_ARGS --limit \$EVAL_SAMPLES_LIMIT"
fi

if [ "\$EVAL_SAMPLES_LOGGING" = true ]; then
    EVAL_EXTRA_ARGS="\$EVAL_EXTRA_ARGS --log_samples"
fi

if [ "\$EVAL_WANDB_LOGGING" = true ]; then
    EVAL_EXTRA_ARGS="\$EVAL_EXTRA_ARGS --wandb_args \$EVAL_WANDB_ARGS"
fi

echo "[info] Schedule \$model on \$task"
source "\$(pwd)"/.venv/bin/activate
python -m accelerate.commands.launch --main_process_port=\$ACCELERATE_MAIN_PROCESS_PORT --num_processes=\$ACCELERATE_NUM_PROCESSES -m eval_model --model "\$model" --tasks "\$task" \$EVAL_EXTRA_ARGS

EOT
}

main "$@"
