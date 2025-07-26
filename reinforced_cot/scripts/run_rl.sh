# ! /bin/bash
# srun -p L40 -J xzm_reinforce_cot -N 1 --ntasks-per-node=1 -w gpu4009 --gres=gpu:l40:3 --cpus-per-task=18 --pty /bin/bash
export TOKENIZERS_PARALLELISM=True

# 自动定位项目根目录并切换
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
PROJECT_ROOT=$(dirname $(dirname "$SCRIPT_DIR"))
cd "$PROJECT_ROOT"
echo "Current working directory has been changed to: $(pwd)"

# 更新所有文件路径为相对于项目根目录的路径
# "configs/fsdp.yaml" "configs/deepspeed2_sft.yaml"
deepspeed_config_file="configs/deepspeed2_sft.yaml"
training_config_file="configs/rl_finetune.yaml"
main_script_path="reinforced_cot/main.py"

training_stage="ppo"
num_processes='4'
main_process_port='8889'

echo "Starting ${training_stage} experiment...."
accelerate launch \
        --config_file "${deepspeed_config_file}" \
        --num_processes=${num_processes} \
        --main_process_port=${main_process_port} \
        "${main_script_path}" \
        --stage ${training_stage} \
        --config_path ${training_config_file}