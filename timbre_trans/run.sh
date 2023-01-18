#!/bin/bash
source ~/.bashrc

cd /MelGAN/
python -u process.py >prolargedatalog_$(date +%m%d%H%M),python -u pre_train.py --input="data/train" >largedatalog_$(date +%m%d%H%M),python -u train.py --input="data/train" >trainlog_$(date +%m%d%H%M)


python -u fast_thoegaze_v4.py --features=melspec --usetrans=0 --pooling_type=gru --dmodel=512 --learning-rate=1e-4 --dropout=0.1 --batch_size=16 --train_nums=20000 --valid_nums=2000 --maxepochs=500 --alpha=1 --beta=1 --gamma=0 --comment=es2ivqmelconstbiggerlr1e4 --nfft=2048 --segwidth=512 --traindata_path=/common-data/liaolin/traindatasets.tfrecord --validdata_path=/common-data/liaolin/validdatasets.tfrecord >trainlog_$(date +%m%d%H%M)


python -u dif_thoegaze.py --features=melspec --usetrans=0 --pooling_type=gru --dmodel=512 --learning-rate=1e-4 --dropout=0.1 --batch_size=8 --train_nums=20000 --valid_nums=2000 --maxepochs=300 --alpha=1 --beta=1 --gamma=0 --comment=diffusion2ivqmelbiggerbetalr1e4 --nfft=2048 --segwidth=512 --traindata_path=/common-data/liaolin/traindatasets.tfrecord --validdata_path=/common-data/liaolin/validdatasets.tfrecord >trainlog_$(date +%m%d%H%M)
