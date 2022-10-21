生成数据需要先安装timitity++

midi数据集：maestro v3 https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip

sf2音色文件：https://sites.google.com/site/soundfonts4u/

Electric-Guitars-JNv4.4.sf2

Acoustic Guitars JNv2.4.sf2

Nice-Strings-PlusOrchestra-v1.6.sf2

Expressive Flute SSO-v1.2.sf2

Chris Mandolin-4U-v3.0.sf2

#!/bin/bash
source ~/.bashrc
cd /data/state-spaces-main/sashimi/
python -u mae_torch_preprocess.py --maestropath=/data/maestro-v3.0.0/ >largedatalog_$(date +%m%d%H%M)
python -u fast_thoegaze_v4.py --usetrans=0 --batch_size=16 --maxepochs=30 --alpha=0.9 --beta=0.1 --gamma=0 --comment=stftconstfirst --nfft=2048 --segwidth=256 --traindata_path=/common-data/liaolin/traindatasets.tfrecord --validdata_path=/common-data/liaolin/validdatasets.tfrecord --dmodel=512 >trainlog_$(date +%m%d%H%M)
