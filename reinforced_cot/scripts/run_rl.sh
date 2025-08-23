# ! /bin/bash
export TOKENIZERS_PARALLELISM=True

# Get the absolute path to the workspace root directory
WS_ROOT=$(dirname $(dirname $(dirname $(realpath $0))))

deepspeed_config_file="$WS_ROOT/configs/deepspeed/deepspeed_grpo.yaml"
training_config_file="$WS_ROOT/configs/train/grpo.yaml"
main_script_path="$WS_ROOT/reinforced_cot/main.py"

training_stage="grpo"
num_processes='2'
main_process_port='8888'

echo "Starting GRPO experiment ...."
accelerate launch \
        --config_file "${deepspeed_config_file}" \
        --num_processes=${num_processes} \
        --main_process_port=${main_process_port} \
        "${main_script_path}"  \
        --stage ${training_stage} \
        --config_path ${training_config_file}