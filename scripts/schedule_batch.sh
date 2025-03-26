#!/usr/bin/env bash
set -o errexit
set -o nounset
set -o pipefail

if [[ "${TRACE-0}" == "1" ]]; then
    set -o xtrace
fi

if [[ "${1-}" =~ ^-*h(elp)?$ ]]; then
    echo 'usage: schedule_batch.sh [-h] [--models MODELS] [--tasks TASKS] [--limit LIMIT]
                                [--model-args ARGS] [--no-samples] [--no-wandb] [--output OUTPUT]

Schedule a batch of runs with the defined models and tasks.

Options:
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
ACCELERATE_MAIN_PROCESS_PORT=$((RANDOM % (50000 - 30000 + 1) + 30000))
ACCELERATE_NUM_PROCESSES=$(
    if [ -n "${CUDA_VISIBLE_DEVICES-}" ]; then
        echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l;
    else
        nvidia-smi --list-gpus | wc -l;
    fi
)

EVAL_MODELS=""
EVAL_MODELS_ARGS=""
EVAL_OUTPUT_DIR=logs/schedule
EVAL_TASKS=""
EVAL_SAMPLES_LIMIT=""
EVAL_SAMPLES_LOGGING=true
EVAL_WANDB_LOGGING=true
EVAL_WANDB_ARGS="project=lmms-owc,job_type=eval"

main() {

    # Parse input arguments
    while [[ $# -gt 0 ]]; do
    case $1 in
        --models) EVAL_MODELS="$2"; shift 2 ;;
        --tasks) EVAL_TASKS="$2"; shift 2 ;;
        --limit) EVAL_SAMPLES_LIMIT="$2"; shift 2 ;;
        --model-args|--models-args) EVAL_MODELS_ARGS="$2"; shift 2 ;;
        --no-samples) EVAL_SAMPLES_LOGGING=false; shift ;;
        --no-wandb) EVAL_WANDB_LOGGING=false; shift ;;
        -o|--output) EVAL_OUTPUT_DIR="$2"; shift 2 ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
        esac
    done

    # Split comma-separated values into array
    IFS=',' read -ra EVAL_MODELS <<< "$EVAL_MODELS"
    IFS=',' read -ra EVAL_TASKS <<< "$EVAL_TASKS"

    if [ "$EVAL_SAMPLES_LOGGING" = false ]; then
        echo "[warn] Skipping samples logging!"
    fi

    if [ "$EVAL_WANDB_LOGGING" = false ]; then
        echo "[warn] Skipping wandb logging!"
    fi

    for task in "${EVAL_TASKS[@]}"; do
        for model in "${EVAL_MODELS[@]}"; do

            EVAL_EXTRA_ARGS=""
            EVAL_EXTRA_ARGS="$EVAL_EXTRA_ARGS --batch_size 1"
            EVAL_EXTRA_ARGS="$EVAL_EXTRA_ARGS --output_path $EVAL_OUTPUT_DIR/$task/$model"

            if [ "$EVAL_MODELS_ARGS" ]; then
                EVAL_EXTRA_ARGS="$EVAL_EXTRA_ARGS --model_args $EVAL_MODELS_ARGS"
            fi

            if [ "$EVAL_SAMPLES_LIMIT" ]; then
                EVAL_EXTRA_ARGS="$EVAL_EXTRA_ARGS --limit $EVAL_SAMPLES_LIMIT"
            fi

            if [ "$EVAL_SAMPLES_LOGGING" = true ]; then
                EVAL_EXTRA_ARGS="$EVAL_EXTRA_ARGS --log_samples"
            fi

            if [ "$EVAL_WANDB_LOGGING" = true ]; then
                EVAL_EXTRA_ARGS="$EVAL_EXTRA_ARGS --wandb_args $EVAL_WANDB_ARGS"
            fi

            # shellcheck source=/dev/null
            source "$(pwd)"/.venv/bin/activate
            echo "[info] Schedule $model on $task"
            # shellcheck disable=SC2048,SC2086
            python -m accelerate.commands.launch \
                --main_process_port="$ACCELERATE_MAIN_PROCESS_PORT" \
                --num_processes="$ACCELERATE_NUM_PROCESSES" \
                -m eval_model --model "$model" --tasks "$task" ${EVAL_EXTRA_ARGS[*]}

        done
    done

}

main "$@"
