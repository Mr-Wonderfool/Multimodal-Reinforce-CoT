# ! /bin/bash
# srun -p L40 -J xzm_reinforce_cot -N 1 --ntasks-per-node=1 -w gpu4009 --gres=gpu:l40:3 --cpus-per-task=18 --pty /bin/bash
export TOKENIZERS_PARALLELISM=True

# Get the absolute path to the workspace root directory
WS_ROOT=$(dirname $(dirname $(dirname $(realpath $0))))

deepspeed_config_file="$WS_ROOT/configs/deepspeed/deepspeed_sft.yaml"
training_config_file="$WS_ROOT/configs/train/sft.yaml"
main_script_path="$WS_ROOT/reinforced_cot/main.py"

training_stage="sft"
num_processes='3'
main_process_port='8888'

echo "Starting sft experiment ...."
accelerate launch \
        --config_file "${deepspeed_config_file}" \
        --num_processes=${num_processes} \
        --main_process_port=${main_process_port} \
        "${main_script_path}"  \
        --stage ${training_stage} \
        --config_path ${training_config_file}