from typing import Any, Callable, Mapping, Optional, Sequence, Tuple, TypeVar, MutableMapping
import librosa
import os
import numpy as np
from scipy.fftpack.basic import fft
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
import dataclasses

import math
from mid2target import mid2target
from tqdm import tqdm
import tfrecord
from torch.utils.data import Dataset
from MelGAN.utils.audio import melspectrogram
import random
import multiprocessing
# from audio import melspectrogram
#token id = time*88+note-21+1 eos=0
from sf2utils.sf2parse import Sf2File # pip install sf2utils
SYNTH_BIN = 'timidity'
import subprocess

from pathlib import Path
import argparse

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# Create a temporary config file pointing to the correct soundfont
parser.add_argument("--samplerate", type=int, default=16000, help='')
parser.add_argument("--nfft", type=int, default=2048, help='')
parser.add_argument("--delete_wav", type=int, default=0, help='')
parser.add_argument("--segwidth", type=int, default=256, help='')
parser.add_argument("--train_nums", type=int, default=100000, help='')
parser.add_argument("--valid_nums", type=int, default=10000, help='')
parser.add_argument("--traindatasets", type=str, default='/data/liaolin/traindatasetsbig', help='')
parser.add_argument("--validdatasets", type=str, default='/data/liaolin/aliddatasetsbig', help='')
parser.add_argument("--maestropath", type=str, default='/data/liaolin/maestro-v3.0.0/', help='')
parser.add_argument("--method", type=str, default='melspec', help='')
args = parser.parse_args()

sample_rate = args.samplerate
hop_width = 200 if args.method == 'melspec' else int(sample_rate/32)
seg_width = args.segwidth
def select_midi_soundfont(name, instrument='default'):
    matches = sorted(Path('/data/liaolin/soundfont/').glob('**/' + name))
    
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
        # print('Using preset', preset_num)
    
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
    # 'piano':('Chateau Grand Lite-v1.0.sf2',''),
    'electric guitar Dry':('Electric-Guitars-JNv4.4.sf2','Clean Guitar GU'),
    # 'electric guitar Distort':('Electric-Guitars-JNv4.4.sf2','Distortion SG'),
    # 'electric guitar Jazz':('Electric-Guitars-JNv4.4.sf2','Jazz Guitar FR3'),
    # 'acoustic guitar':('Acoustic Guitars JNv2.4.sf2',''),
    # 'string':('Nice-Strings-PlusOrchestra-v1.6.sf2','String'),
    'orchestra':('Nice-Strings-PlusOrchestra-v1.6.sf2','Orchestra'),
    # 'cello':('Nice-Strings-PlusOrchestra-v1.6.sf2','Cello 1'),
    # 'violin':('Nice-Strings-PlusOrchestra-v1.6.sf2','Violin 1'),
    # 'brass':('Nice-Strings-PlusOrchestra-v1.6.sf2','Brass'),
    # 'trumpet':('Nice-Strings-PlusOrchestra-v1.6.sf2','Trumpet 2'),
    # 'flute':('Expressive Flute SSO-v1.2.sf2',''),
    # 'mandolin':('Chris Mandolin-4U-v3.0.sf2','Full Exp Mandolin'),
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
def renderMidi( f, cfgs,instrument,return_file_path=False):
    # print(str(f), flush=True)
    # min_len = 2 << 63
    # render the waveform files
    # for (j, instrument) in enumerate(instruments):
    # print(instrument, '...')
    # synthesize midi with timidity++, obtain waveform
    file = Path(f).with_suffix('.'+instrument.replace(' ', '-')+'.wav')
    if midi2wav(f, file, cfgs)!=0:
        print('midi2wav failed!')
        return 0
    else:
        if return_file_path:
            return file
        else:
            return 1

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


fmin = 40
min_level_db = -100
ref_level_db = 20

def amp_to_db(x):
    return 20 * np.log10(np.maximum(1e-5, x))

def normalize(S):
    return np.clip((S - min_level_db) / -min_level_db, 0, 1)

def tokenize(midfile=None, audio=None,method='cqt',return_target=True,delete_wav=args.delete_wav,sample_rate=sample_rate,hop_width=hop_width,nfft=args.nfft,seg_width=seg_width,return_padding_length=False):

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
        frames = librosa.stft(y=frames,n_fft=nfft, hop_length=hop_width)
        # frames,phase=librosa.magphase(frames)
        frames = amp_to_db(np.abs(frames)) - ref_level_db
        # frames=np.log1p(frames)
        frames = normalize(frames)
    elif method == 'melspec':
        # frames = librosa.feature.melspectrogram(y=frames, sr=sample_rate, n_fft=2048, hop_length=hop_width,n_mels=229, fmin=30, fmax=8000)# librosa.feature.melspectrogram(audio, sr=16000, n_fft=2048, hop_length=160, n_mels=229, fmin=30, fmax=8000)
        frames = melspectrogram(frames).astype(np.float32)
        frames=np.abs(frames)

    # frames = np.abs(frames)
    frames = np.transpose(frames)
    temp, nbins = frames.shape
    # print("nbins",nbins)
    padding_length=seg_width-temp%seg_width
    frames = np.pad(frames, ((0, padding_length), (0, 0)))

    # if onsets_only:
    #   times, values = note_sequence_to_onsets(ns)
    # else:
    #   ns = note_seq.apply_sustain_control_changes(ns)
    #   times, values = (note_sequence_to_onsets_and_offsets_and_programs(ns))

    audio_split = np.reshape(frames, [-1, seg_width, nbins])
    if delete_wav:
        print('delete',audio)
        os.remove(audio)
    if return_target:
        segs_num = audio_split.shape[0]
        # targets = np.array(dump_targets(midfile, segs_num),dtype=object)
        targets = dump_targets(midfile, segs_num)

        return targets, audio_split, segs_num
    else:
        if return_padding_length:
            return  audio_split,padding_length
        else:
            return audio_split


def dump_targets(midfile, segs_num):

    targets = mid2target(midfile,
                      seg_width, hop_width,sample_rate)
    # print(segs_num,type(segs_num))
    for _ in range(segs_num-len(targets)):
      targets.append([[] for _ in range(seg_width)])
    
    assert len(targets) == segs_num

    return targets
   
def remove_wav(path):
    try:
        os.remove(path)
    except:
        pass


def make_datasets(path, output_file,voutput_file,tag='train',nums=None,vnums=None,render=True):
    
    mid_Filelist = []
    for home, dirs, files in os.walk(path):
        for filename in files:
            # if '2018' in filename:
            # 文件名列表，包含完整路径
                if tag=='train':
                    if 'midi' == filename[-4:]:
                        mid_Filelist.append(os.path.join(home, filename))
                # else:
                #     if 'midi' == filename[-4:] and '3.midi'  in filename:
                #         mid_Filelist.append(os.path.join(home, filename))
            # # 文件名列表，只包含文件名
            # Filelist.append( filename)

    writer=tfrecord.TFRecordWriter(output_file)
    vwriter=tfrecord.TFRecordWriter(voutput_file)
    cout=0
    vout=0
    mode=0
    render_failed=0
    random.shuffle(mid_Filelist)
    inst_num=len(instruments)
    inst_list=list(instruments)
    for instrument in instruments:
        renderMidi(mid_Filelist[0], select_midi_soundfont(
            *instruments[instrument]), instrument)
        renderMidi(mid_Filelist[1], select_midi_soundfont(
            *instruments[instrument]), instrument)
    print('midi nums:',len(mid_Filelist))
    for j in range(1,len(mid_Filelist)//2):

        random.shuffle(inst_list)
        file1=mid_Filelist[j*2-2]
        file2=mid_Filelist[j*2+1-2]
        file3 = mid_Filelist[j*2]
        file4 = mid_Filelist[j*2+1]
        # if render:
        #     for instrument in instruments:
        #         renderMidi( file1, select_midi_soundfont(*instruments[instrument]),instrument)
        #         renderMidi( file2, select_midi_soundfont(*instruments[instrument]),instrument)
        for i in range(inst_num//2):
            

            try:
 
                p = multiprocessing.Pool(processes=8)
                if not render_failed:

                    r0 = p.apply_async(tokenize,(file1, file1[:-4]+inst_list[i*2].replace(' ','-')+'.wav',args.method,))
                    r1 = p.apply_async(
                        tokenize, (file2, file2[:-4]+inst_list[i*2].replace(' ', '-')+'.wav', args.method,))
                    r2 = p.apply_async(tokenize,(
                        file1, file1[:-4]+inst_list[i*2+1].replace(' ', '-')+'.wav', args.method, False,))
                    r3 = p.apply_async(tokenize, (file2, file2[:-4]+inst_list[i*2+1].replace(
                        ' ', '-')+'.wav', args.method, False,))
                result=[]
                if render and vnums != vout:

                    for instrument in instruments:
                        result.append(p.apply_async( renderMidi,(file3, select_midi_soundfont(
                            *instruments[instrument]), instrument,)))
                        
                        result.append ( p.apply_async(renderMidi, (file4, select_midi_soundfont(
                            *instruments[instrument]), instrument,)))

                p.close()
                p.join()

                remove_wav(file1[:-4]+inst_list[i*2].replace(' ', '-')+'.wav')
                remove_wav(file2[:-4]+inst_list[i*2].replace(' ', '-')+'.wav')
                remove_wav(file1[:-4]+inst_list[i*2+1].replace(' ', '-')+'.wav')
                remove_wav(file2[:-4]+inst_list[i*2+1].replace(' ', '-')+'.wav')

                if not render_failed:
                    targets0, split_audio0, _ =r0.get()
                    targets1, split_audio1, _ =r1.get()
                    split_audio2=r2.get()
                    split_audio3=r3.get()


                    z0=[]#list(zip(targets0, list(split_audio0),list(split_audio2)))
                    z1=[]#list(zip(targets1, list(split_audio1),list(split_audio3)))

                    for t0, s0 ,s2 in zip(targets0, list(split_audio0),list(split_audio2)):

                        # if np.all(t0==0): #and random.randint(0, 100)!=5: 
                        if t0==[[]]*seg_width: 
                            continue
                        z0.append((t0,s0,s2))
                    for t1, s1 ,s3 in zip(targets1, list(split_audio1),list(split_audio3)):
                        # print(178,s.shape)
                        if t1==[[]]*seg_width: #and random.randint(0, 100)!=5:  
                            continue
                        z1.append((t1,s1,s3))
                    random.shuffle(z1)
                    random.shuffle(z0)                
                    for j in range(min(len(z1),len(z0))):
                        t0, s0, s2 = z0[j]
                        t1, s1 ,s3 = z1[j]
                        # print(246,t0,t1)
                        if  mode == 0:
                            
                            writer.write({
                                'x0': (s0.reshape([-1]).tobytes(), 'byte'), 
                                'x1': (s1.reshape([-1]).tobytes(), 'byte'),
                                'x2': (s2.reshape([-1]).tobytes(), 'byte'),
                                'x3': (s3.reshape([-1]).tobytes(), 'byte'),
                                't0':(str(t0).encode('utf-8'), 'byte'),
                                't1':(str(t1).encode('utf-8'), 'byte'),
                            })
                            cout+=1
                            print(cout, '/', nums)
                            if nums:
                                if cout==nums:
                                    writer.close()
                                    mode=1
                        if mode==1:
                            vwriter.write({
                                'x0': (s0.reshape([-1]).tobytes(), 'byte'), 
                                'x1': (s1.reshape([-1]).tobytes(), 'byte'),
                                'x2': (s2.reshape([-1]).tobytes(), 'byte'),
                                'x3': (s3.reshape([-1]).tobytes(), 'byte'),
                                't0':(str(t0).encode('utf-8'), 'byte'),
                                't1':(str(t1).encode('utf-8'), 'byte'),
                            })
                            vout+=1
                            print(vout,'/',vnums)
                            if vnums:
                                if vout==vnums:
                                    vwriter.close()
                                    return cout,vout
                render_failed=0
                for r in result:
                    if not r.get():
                        render_failed=1
                        break

            except AssertionError as e: 
                # print(file,'too short <10s') 
                continue
    writer.close()

    return cout,vout

def find_files(root):
    for d, dirs, files in os.walk(root):
        for f in files:
            yield os.path.join(d, f)

if __name__ == "__main__":
    path = args.maestropath
    # stage1 RENDER MIDI TO WAV, you need to makdir ./sf2, and move the .sf2 file into it.

    # for home, dirs, files in os.walk(path):
    #     for filename in files:
    #         # 文件名列表，包含完整路径
    #         if 'midi' == filename[-4:] and 'byte' not in filename:
    #             # mid_Filelist.append(os.path.join(home, filename))
    #             for instrument in instruments:
                    
    #                 dataFile = os.path.join(home, filename[:-4])+instrument+'.wav'
    #                 if os.path.exists(dataFile):
    #                     print(str(dataFile), 'already exists!')
    #                     continue
   
    #                 else:
    #                     renderMidi( os.path.join(home, filename), select_midi_soundfont(*instruments[instrument]),instrument) 
                

    # stage2 make tfrecord dataset, need to do it for twice, one for training ,one for evaluating, maestrov3 except 2018 is used as traindata,2018 for valid. so you need to edit the hierarchy of folders of maestrov3

    output_file =args.traindatasets+'.tfrecord'
    voutput_file =args.validdatasets+'.tfrecord'
    cout,vcout = make_datasets(path, output_file,voutput_file,tag='train',nums=args.train_nums,vnums=args.valid_nums)

    print("train_cout", cout)
    # 0805:train：429405 valid:72055
    
    # cout = make_datasets(path, output_file,tag='valid',nums=args.valid_nums)

    print("valid_cout", vcout)
