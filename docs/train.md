# Prerequisites
1. All configurations are based under `Multimodal-Reinforce-CoT/configs`, simply change parameter settings in `yaml` files and never have to worry about command line arguments. 
2. All training scripts are placed under `Multimodal-Reinforce-CoT/reinforced_cot/scripts`, we use accelerate to launch the experiment.
3. We conducted the experiment using 2 Nvidia A100 GPUs, training took around 7 hours for GRPO.
4. We utilize [deepspeed](https://github.com/deepspeedai/DeepSpeed) for distributed training.

# Train SFT Model
1. SFT weights can be trained with:
```bash
cd Multimodal-Reinforce-CoT/reinforce_cot/scripts
chmod 777 run_sft.sh
./run_sft.sh
```
2. Every experiment will generate a logger directory that looks like the following:
```bash
├─20250824_161306_SFT
   ├─backup
   ├─ckpt
   ├─result
   ├─tensorboard
   └─output.log
```
where `ckpt` stores the saved model weights during experiments, and result directory stores the evaluation output (in json format). Thus you can find the saved SFT weights under `ckpt` directory (for SFT only).

# Train GRPO/GSPO Model
1. Before reinforcement training, merge SFT LoRA adapters into the base model. You will need to change `main(args)` to `merge_weights(args)` in `Multimodal-Reinforce-CoT/reinforced_cot/main.py`, and specify the output directory in function `merge_weights`. Then start the merge process with `run_sft.sh`. For the following instructions, we will assume the merged weights are saved under `qwen/qwen-merged-sft/`.
2. Change `merge_weights(args)` back to `main(args)`, and launch the training with:
```bash
cd Multimodal-Reinforce-CoT/reinforce_cot/scripts
chmod 777 run_rl.sh
./run_rl.sh
```
3. We rely on [HuggingFace Trl](https://github.com/huggingface/trl) for the algorithm, and [vllm](https://github.com/vllm-project/vllm) to speed up generation. The configuration file `Multimodal-Reinforce-CoT/configs/train/grpo.yaml` is a wrap around HuggingFace `TrainingArguments` and `GRPOConfig`. When you wish to run GSPO, simply change `importance_sampling_level` from `token` to `sequence`. One important thing to notice is that **quantization is incompatible with vllm**, so you cannot set both `quantization` and `use_vllm` to `True`.
4. The model weights will be saved under `output_dir` specified in `grpo.yaml`. 