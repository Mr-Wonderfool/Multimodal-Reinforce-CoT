# ! /bin/bash
# conda create -n xzm-qwen-finetune python=3.10 -y
# conda activate xzm-qwen-finetune

# set up python packages
echo "----- installing python packages -----"
pip install -r requirements.txt
echo "----- setting up torch -----"
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121
# conda install pytorch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 pytorch-cuda=12.1 -c pytorch -c nvidia
# set up custom packages
echo "----- Setting up project package -----"
pip install -e .