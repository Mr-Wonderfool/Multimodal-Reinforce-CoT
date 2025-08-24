# ! /bin/bash

# set up python packages
echo "----- setting up torch -----"
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu126
echo "----- installing python packages -----"
pip install -r requirements.txt
# set up custom packages
echo "----- Setting up project package -----"
pip install -e .