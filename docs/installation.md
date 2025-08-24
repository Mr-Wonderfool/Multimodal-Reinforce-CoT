# Installation Instructions
1. Create a conda/python virtual environment and activate it. 
```bash
conda create -n llm-finetune python=3.10 -y
conda activate llm-finetune
```
2. Install related packages, we provide a setup script to simplify this process:
```bash
chmod 777 setup.sh
./setup.sh
```
3. Pull Qwen2.5-VL-3B-Instruct, for which we have also provided a helper script:
```bash
python pull_model.py
```
4. Prepare the dataset, simply unzip `dataset.zip` and place it on the same level with the repository.
5. The final project structure should look like the following:
```bash
├─Multimodal-Reinforce-CoT
├─qwen
│  └─Qwen2.5-VL-3B-Instruct
├─data
│  └─multimodal-cot
```