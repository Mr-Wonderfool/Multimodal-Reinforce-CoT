# ! /bin/bash
# conda create -n xzm-qwen-finetune python=3.10 -y
# conda activate xzm-qwen-finetune

# set up python packages
echo "----- installing python packages -----"
pip install -r requirements.txt
echo "----- setting up torch -----"
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=11.8 -c pytorch -c nvidia
# set up custom packages
echo "----- Setting up project package -----"
pip install -e .