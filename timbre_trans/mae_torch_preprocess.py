from typing import Any, Callable, Mapping, Optional, Sequence, Tuple, TypeVar, MutableMapping
import librosa
import os
import numpy as np
from scipy.fftpack.basic import fft
# import tensorflow as tf
# import note_seq
import dataclasses
# import event_codec
# import note_sequences
# import vocabularies
import math
from mid2target import mid2target
from tqdm import tqdm
import tfrecord
from torch.utils.data import Dataset
# import seqio
# from mt3 import event_codec, run_length_encoding, note_sequences, vocabularies
# from mt3.vocabularies import build_codec
import random
#token id = time*88+note-21+1 eos=0
from sf2utils.sf2parse import Sf2File # pip install sf2utils
SYNTH_BIN = 'timidity'
import subprocess
hop_width = 256
seg_width = 64
sample_rate = 12800
from pathlib import Path

# Create a temporary config file pointing to the correct soundfont
def select_midi_soundfont(name, instrument='default'):
    matches = sorted(Path('./data/soundfont/').glob('**/' + name))
    matches = sorted(Path('./sf2').glob('**/' + name))
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
    with open(cfgpath, 'w') as f:
        config = "dir {}\nbank 0\n0 %font \"{}\" 0 {} amp=100".format(fontpath.parent.resolve(), name, preset_num)
        f.write(config)
    return cfgpath

# render matching audio for each of these soundfonts
instruments = {
    # #'piano': ('grand-piano-YDP-20160804.sf2', ''),
    # 'piano': ('SoundBlasterPiano.sf2', ''),
    # # 'flute': ('Milton_Pan_flute.sf2', ''),
    # 'guitar': ('spanish-classical-guitar.sf2', ''),
    # # 'harp' : ('Roland_SC-88.sf2', 'Harp'),
    # # 'kalimba' : ('Roland_SC-88.sf2', 'Kalimba'),
    # # 'pan' : ('Roland_SC-88.sf2', 'Pan flute')
    'piano':('Chateau Grand Lite-v1.0.sf2',''),
    'electric guitar':('Electric-Guitars-JNv4.4.sf2',''),
    'acoustic guitar':('Acoustic Guitars JNv2.4.sf2',''),
    'string':('Nice-Strings-PlusOrchestra-v1.6.sf2','String'),
    'orchestra':('Nice-Strings-PlusOrchestra-v1.6.sf2','Orchestra'),
    'flute':('Expressive Flute SSO-v1.2.sf2','')
}
def midi2wav(file, outpath, cfg):
    # print(71,cfg)
    cmds = [
        SYNTH_BIN,'-s',str(sample_rate), '-c', str(cfg), str(file),
        '-Od', '--reverb=g,25' '--noise-shaping=4'
        '-EwpvseToz', '-f', '-A100', '-Ow',
        '-o', str(outpath)
    ]
    # print(78,' '.join(cmds))
    #print('Converting midi to wav...', end='', flush=True)
    return subprocess.call(cmds, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
# render the given midi file with the given instruments, generate spectrograms and save
def renderMidi( f, cfgs,instrument):
    print(str(f), flush=True)
    # min_len = 2 << 63
    # render the waveform files
    # for (j, instrument) in enumerate(instruments):
    print(instrument, '...')
    # synthesize midi with timidity++, obtain waveform
    file = Path(f).with_suffix('.'+instrument+'.wav')
    if midi2wav(f, file, cfgs)!=0:
        print('midi2wav failed!')
        return
    # cur_len = len(librosa.load(str(file), sr = sample_rate)[0])
    # min_len = min(min_len, cur_len)




@dataclasses.dataclass
class NoteEncodingState:
  """Encoding state for note transcription, keeping track of active pitches."""
  # velocity bin for active pitches and programs
  active_pitches: MutableMapping[Tuple[int, int], int] = dataclasses.field(
      default_factory=dict)




def _audio_to_frames(
    samples: Sequence[float], hop_width
) -> Tuple[Sequence[Sequence[int]], np.ndarray]:
  """Convert audio samples to non-overlapping frames and frame times."""
    # hop_with：一个hop几个samples

  samples = np.pad(samples,
                   [0, hop_width - len(samples) % hop_width],
                   mode='constant')  # 将最后不够hopsize的部分补全,和split_audio功能重复。可删除。


  return samples


def load_audio(audio):
    samples, _ = librosa.load(audio, sr=sample_rate)

    return samples








def tokenize(midfile, audio,method='cqt',return_target=True):

    # note_sequences.validate_note_sequence(ns)
    samples = load_audio(audio)
    frames = _audio_to_frames(samples,hop_width)
    # print (88,frames.shape)
    assert frames.shape[0] >= sample_rate*10  # 过短10s以下被舍弃
    # print(183, frames.shape)
    # frames = np.reshape(frames, [-1])
    # frames = frames[:256*16]
    if method=='cqt':
        frames = librosa.cqt(frames, sr=sample_rate,
                         hop_length=hop_width, fmin=27.50, n_bins=nbins, bins_per_octave=36)
    elif method == 'stft':
        frames = librosa.stft(y=frames,n_fft=512, hop_length=hop_width)
    elif method == 'melspec':
        frames = librosa.feature.melspectrogram(y=frames, sr=sample_rate, n_fft=256, hop_length=256,n_mels=128)
    frames = np.abs(frames)
    frames = np.transpose(frames)
    temp, nbins = frames.shape
    # print("nbins",nbins)
    frames = np.pad(frames, ((0, seg_width-temp % seg_width), (0, 0)))
    # print(191,frames.shape)
    # if onsets_only:
    #   times, values = note_sequence_to_onsets(ns)
    # else:
    #   ns = note_seq.apply_sustain_control_changes(ns)
    #   times, values = (note_sequence_to_onsets_and_offsets_and_programs(ns))

    audio_split = np.reshape(frames, [-1, seg_width, nbins])
    
    if return_target:
        segs_num = audio_split.shape[0]
        targets = np.array(dump_targets(midfile, segs_num),dtype=object)

        return targets, audio_split, segs_num
    else:
        return  audio_split


def dump_targets(midfile, segs_num):

    targets = mid2target(midfile,
                      seg_width, hop_width,sample_rate)
    # print(segs_num,type(segs_num))
    for _ in range(segs_num-len(targets)):
      targets.append([[0]*88]*seg_width)
    
    assert len(targets) == segs_num

    return targets
   



def make_datasets(path, output_file):
    
    mid_Filelist = []
    for home, dirs, files in os.walk(path):
        for filename in files:
            # 文件名列表，包含完整路径
            if 'midi' == filename[-4:] and 'byte' not in filename:
                mid_Filelist.append(os.path.join(home, filename))
            # # 文件名列表，只包含文件名
            # Filelist.append( filename)

    writer=tfrecord.TFRecordWriter(output_file)
    cout=0
    random.shuffle(mid_Filelist)
    inst_num=len(instruments)
    inst_list=list(instruments)

    for j in range(len(mid_Filelist)//2):

        random.shuffle(inst_list)
        file1=mid_Filelist[j*2]
        file2=mid_Filelist[j*2+1]
        for i in range(inst_num//2):


            try:
                # random.shuffle(instruments)
                    # print(i,file1,instruments)
                    targets0, split_audio0, _ = tokenize(file1, file1[:-4]+inst_list[i*2]+'.wav',method='stft')
                    targets1, split_audio1, _ = tokenize(file1, file1[:-4]+inst_list[i*2+1]+'.wav',method='stft')
                    split_audio2 = tokenize(file1, file1[:-4]+inst_list[i*2+1]+'.wav',method='stft',return_target=False)
                    split_audio3 = tokenize(file1, file2[:-4]+inst_list[i*2+1]+'.wav',method='stft',return_target=False)


                    z0=[]#list(zip(targets0, list(split_audio0),list(split_audio2)))
                    z1=[]#list(zip(targets1, list(split_audio1),list(split_audio3)))

                    for t0, s0 ,s2 in zip(targets0, list(split_audio0),list(split_audio2)):

                        if np.all(t0==0): #and random.randint(0, 100)!=5:  
                            continue
                        z0.append((t0,s0,s2))
                    for t1, s1 ,s3 in zip(targets0, list(split_audio1),list(split_audio3)):
                        # print(178,s.shape)
                        if np.all(t1==0): #and random.randint(0, 100)!=5:  
                            continue
                        z1.append((t1,s1,s3))
                    random.shuffle(z1)
                    random.shuffle(z0)                
                    for j in range(min(len(z1),len(z0))):
                        t0, s0, s2 = z0[j]
                        t1, s1 ,s3 = z1[j]
                        writer.write({
                            'x0': (s0.reshape([-1]).tobytes(), 'byte'),
                            'x1': (s1.reshape([-1]).tobytes(), 'byte'),
                            'x2': (s2.reshape([-1]).tobytes(), 'byte'),
                            'x3': (s3.reshape([-1]).tobytes(), 'byte'),
                            't0':(t0.reshape([-1]).tobytes(), 'byte'),
                            't1':(t1.reshape([-1]).tobytes(), 'byte'),
                        })
                        cout+=1
            except AssertionError as e: 
                # print(file,'too short <10s') 
                continue
    writer.close()

    return cout
# class TTDataset(Dataset):


#     def __init__(self, folder, transform=None):
#         # all_classes = classes
#         class_to_idx = {classes[i]: i for i in range(len(classes))}

#         data = []
#         # for c in all_classes:
#         #     d = os.path.join(folder, c)
#         #     target = class_to_idx[c]
#         #     for f in os.listdir(d):
#         #         path = os.path.join(d, f)
#         #         data.append((path, target))

#         # self.classes = classes
#         self.data = data
#         self.transform = transform

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, index):
#         path, target = self.data[index]
#         data = {'path': path, 'target': target}

#         if self.transform is not None:
#             data = self.transform(data)

#         return data

#     def make_weights_for_balanced_classes(self):
#         """adopted from https://discuss.pytorch.org/t/balanced-sampling-between-classes-with-torchvision-dataloader/2703/3"""

#         nclasses = len(self.classes)
#         count = np.zeros(nclasses)
#         for item in self.data:
#             count[item[1]] += 1

#         N = float(sum(count))
#         weight_per_class = N / count
#         weight = np.zeros(len(self))
#         for idx, item in enumerate(self.data):
#             weight[idx] = weight_per_class[item[1]]
#         return weight

# helper: find all possible files in a certain directory (and subdirectories)
def find_files(root):
    for d, dirs, files in os.walk(root):
        for f in files:
            yield os.path.join(d, f)

if __name__ == "__main__":
    path = './maestro-v3.0.0'
    # RENDER MIDI TO WAV
    cfgs = []
    os.makedirs('data/waves', exist_ok=True)
    for home, dirs, files in os.walk(path):
        for filename in files:
            # 文件名列表，包含完整路径
            if 'midi' == filename[-4:] and 'byte' not in filename:
                # mid_Filelist.append(os.path.join(home, filename))
                for instrument in instruments:
                    
                    dataFile = os.path.join(home, filename[:-4])+instrument+'.wav'
                    if os.path.exists(dataFile):
                        print(str(dataFile), 'already exists!')
                        continue
   
                    else:
                        renderMidi( os.path.join(home, filename), select_midi_soundfont(*instruments[instrument]),instrument) 
                











    # ns = midi2noteseq(midifile)
    # ds=tokenize(ns,audio)
    # v, s = tokenize(midifile, audio)
    
    # path = '/mnt/data/piano/workspace/midi/original'
    output_file ='mae_timbre_small0805.tfrecord'
    cout = make_datasets(path, output_file)
    # from tf_record_split import split_record
    # split_record(output_file)
    # from train_dev_split import merge_record
    # merge_record()
    print("cout", cout)
