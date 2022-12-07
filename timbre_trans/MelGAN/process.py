import pickle
import os
import glob
import argparse
import numpy as np
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from utils.audio import convert_audio, hop_length, sample_rate
from tqdm import tqdm
import random
import shutil
train_rate = 0.95
test_rate  = 0.05

def find_files(path, pattren="*.wav"):
    filenames = []
    for filename in glob.iglob(f'{path}/**/*{pattren}', recursive=True):
       filenames.append(filename)
    return filenames

def data_prepare(audio_path, mel_path, wav_file):
    mel, audio = convert_audio(wav_file)
    np.save(audio_path, audio, allow_pickle=False)
    np.save(mel_path, mel, allow_pickle=False)
    # os.remove(wav_file)
    return audio_path, mel_path, mel.shape[0]

def process(output_dir, wav_files, train_dir, test_dir,  num_workers):
    executor = ProcessPoolExecutor(max_workers=num_workers)
    results = []
    names = []

    random.shuffle(wav_files)
    train_num = int(len(wav_files) * train_rate)

    for wav_file in wav_files[0 : train_num]:
        fid = os.path.basename(wav_file).replace('.wav','.npy')
        names.append(fid)
        # results.append(executor.submit(partial(data_prepare, os.path.join(train_dir, "audio", fid), os.path.join(train_dir, "mel", fid), wav_file)))
        results.append(data_prepare(os.path.join(train_dir, "audio", fid), os.path.join(train_dir, "mel", fid), wav_file))
        # os.remove(wav_file)
    with open(os.path.join(output_dir, "train", 'names.pkl'), 'wb') as f:
        pickle.dump(names, f)

    names = []
    for wav_file in wav_files[train_num : len(wav_files)]:
        fid = os.path.basename(wav_file).replace('.wav','.npy')
        names.append(fid)
        # results.append(executor.submit(partial(data_prepare, os.path.join(test_dir, "audio", fid), os.path.join(test_dir, "mel", fid), wav_file)))
        results.append(data_prepare(os.path.join(test_dir, "audio", fid), os.path.join(test_dir, "mel", fid), wav_file))
        # os.remove(wav_file)
    with open(os.path.join(output_dir, "test", 'names.pkl'), 'wb') as f:
        pickle.dump(names, f)


    return results #[result.result() for result in tqdm(results)]

def preprocess(args):
    train_dir = os.path.join(args.output, 'train')
    test_dir = os.path.join(args.output, 'test')
    os.makedirs(args.output, exist_ok=True)
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    os.makedirs(os.path.join(train_dir, "audio"), exist_ok=True)
    os.makedirs(os.path.join(train_dir, "mel"), exist_ok=True)
    os.makedirs(os.path.join(test_dir, "audio"), exist_ok=True)
    os.makedirs(os.path.join(test_dir, "mel"), exist_ok=True)

    wav_files = find_files(args.wav_dir)
    metadata = process(args.output, wav_files, train_dir, test_dir, args.num_workers)
    write_metadata(metadata, args.output)
    # import os
    
    # shutil.rmtree(args.wav_dir)

def write_metadata(metadata, out_dir):
    with open(os.path.join(out_dir, 'metadata.txt'), 'w', encoding='utf-8') as f:
        for m in metadata:
            f.write('|'.join([str(x) for x in m]) + '\n')
    frames = sum([m[2] for m in metadata])
    frame_shift_ms = hop_length * 1000 / sample_rate
    hours = frames * frame_shift_ms / (3600 * 1000)
    print('Write %d utterances, %d frames (%.2f hours)' % (len(metadata), frames, hours))
SYNTH_BIN = 'timidity'
from pathlib import Path
from sf2utils.sf2parse import Sf2File
def select_midi_soundfont(name, instrument='default'):
    # matches = sorted(Path('./data/soundfont/').glob('**/' + name))
    matches = sorted(Path('/data/state-spaces-main/sashimi/sf2/').glob('**/' + name))
    if len(matches) == 0:
        raise Exception('Could not find soundfont: ' + name)
    elif len(matches) > 1:
        print('Multiple matching soundfonts:', matches)
        print('Using first match')
    fontpath = matches[0]

    with open(fontpath.resolve(), 'rb') as sf2_file:
        sf2 = Sf2File(sf2_file)
        preset_num = sf2.presets[0].preset
        for preset in sf2.presets:
            if preset.name.lower() == instrument.lower():
                preset_num = preset.preset
            #if preset.name != 'EOP':
            #    print('Preset {}: {}'.format(preset.preset, preset.name))
        print('Using preset', preset_num)
    
    cfgpath = fontpath.with_suffix('.'+instrument+'.cfg')
    print(107,cfgpath)
    with open(cfgpath, 'w') as f:
        config = "dir {}\nbank 0\n0 %font \"{}\" 0 {} amp=100".format(fontpath.parent.resolve(), name, preset_num)
        f.write(config)
    return cfgpath
instruments = {
    # #'piano': ('grand-piano-YDP-20160804.sf2', ''),
    # 'piano': ('SoundBlasterPiano.sf2', ''),
    # # 'flute': ('Milton_Pan_flute.sf2', ''),
    # 'guitar': ('spanish-classical-guitar.sf2', ''),
    # # 'harp' : ('Roland_SC-88.sf2', 'Harp'),
    # # 'kalimba' : ('Roland_SC-88.sf2', 'Kalimba'),
    # # 'pan' : ('Roland_SC-88.sf2', 'Pan flute')
    # 'piano':('Chateau Grand Lite-v1.0.sf2',''),
    'electric guitar Dry':('Electric-Guitars-JNv4.4.sf2','Clean Guitar GU'),
    'electric guitar Distort':('Electric-Guitars-JNv4.4.sf2','Distortion SG'),
    'electric guitar Jazz':('Electric-Guitars-JNv4.4.sf2','Jazz Guitar FR3'),
    'acoustic guitar':('Acoustic Guitars JNv2.4.sf2',''),
    'string':('Nice-Strings-PlusOrchestra-v1.6.sf2','String'),
    'orchestra':('Nice-Strings-PlusOrchestra-v1.6.sf2','Orchestra'),
    'cello':('Nice-Strings-PlusOrchestra-v1.6.sf2','Cello 1'),
    'violin':('Nice-Strings-PlusOrchestra-v1.6.sf2','Violin 1'),
    'brass':('Nice-Strings-PlusOrchestra-v1.6.sf2','Brass'),
    'trumpet':('Nice-Strings-PlusOrchestra-v1.6.sf2','Trumpet 2'),
    'flute':('Expressive Flute SSO-v1.2.sf2',''),
    'mandolin':('Chris Mandolin-4U-v3.0.sf2','Full Exp Mandolin'),
}


import subprocess
def midi2wav(file, outpath, cfg):
    # print(71,cfg)
    cmds = [
        SYNTH_BIN,'-s',str(sample_rate), '-c', str(cfg), str(file),
        '-Od', '--reverb=g,25' '--noise-shaping=4'
        '-EwpvseToz', '-f', '-A100', '-Ow',
        '-o', str(outpath)
    ]
    print(145,cmds)
    # print(78,' '.join(cmds))
    #print('Converting midi to wav...', end='', flush=True)
    return subprocess.call(cmds, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
def renderMidi( f, cfgs,instrument,return_file_path=False,index=0,args=None):
    print(str(f), flush=True)
    # min_len = 2 << 63
    # render the waveform files
    # for (j, instrument) in enumerate(instruments):
    print(instrument, '...')
    # synthesize midi with timidity++, obtain waveform
    file = args.wav_dir+str(index)+'.'+instrument+'.wav'
    if midi2wav(f, file, cfgs)!=0:
        print('midi2wav failed!')
    else:
        if return_file_path:
            return file
def render(path,args):
    mid_Filelist = []
    for home, dirs, files in os.walk(path):
        for filename in files:
            # 文件名列表，包含完整路径
            # if tag=='train':
            #     if 'midi' == filename[-4:] and '2018' not in filename:
            #         mid_Filelist.append(os.path.join(home, filename))
            # else:
            if 'midi' == filename[-4:] and '2018'  in filename:
                mid_Filelist.append(os.path.join(home, filename))
    for i,file1 in enumerate(mid_Filelist[:100]):
        for instrument in instruments:
            renderMidi( file1, select_midi_soundfont(*instruments[instrument]),instrument,index=i,args=args)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wav_dir', default='wavs/')
    parser.add_argument('--output', default='data')
    parser.add_argument('--num_workers', type=int, default=8)
    args = parser.parse_args()
    # render('/common-data/liaolin/maestro-v3.0.0/',args)
    preprocess(args)

if __name__ == "__main__":
    main()
