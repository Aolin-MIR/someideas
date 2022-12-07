#!/bin/bash
source ~/.bashrc
cd /data/FB-MelGAN-master/
python -u   >prolargedatalog_$(date +%m%d%H%M)
python -u pre_train.py --input="data/train" >largedatalog_$(date +%m%d%H%M)
python -u train.py --input="data/train" >trainlog_$(date +%m%d%H%M)