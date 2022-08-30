#!/bin/bash
source ~/.bashrc
cd /data/state-spaces-main/sashimi/
python -u fast_thoegaze_v3.py --batch_size=32 --beta=0.7 --gamma=0.3 --train_nums=183347 --valid_nums=32504 >trainlog_$(date +%m%d%H%M)
