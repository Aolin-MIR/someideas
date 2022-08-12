生成数据需要先安装timitity++
midi数据集：maestro v3 https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip
sf2音色文件：https://sites.google.com/site/soundfonts4u/

Electric-Guitars-JNv4.4.sf2
Acoustic Guitars JNv2.4.sf2
Nice-Strings-PlusOrchestra-v1.6.sf2
Expressive Flute SSO-v1.2.sf2
Chris Mandolin-4U-v3.0.sf2

数据处理 ： python mae_torch_preprocess.py --nfft=2048 --samplerate=25600 --delete_wav=1 --traindatasets=PATH_TO_TRAIN_DATASETS --validdatasets=PATH_TOVLID_DATASETS --maestropath=PATH_TO_MAETRO_DATASETS


训练： python fast_thoegaze.py --nfft=2048  --dmodel=768 --traindatasets=PATH_TO_TRAIN_DATASETS --validdatasets=PATH_TOVLID_DATASETS --batch_size=128 --segwidth=256 --train_nums=xxx --valid_nums=xxx
