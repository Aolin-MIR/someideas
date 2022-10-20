import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import gc
import os
from einops import rearrange 
import torch
import copy
from mae_torch_preprocess import tokenize
import scipy.signal as signal
import argparse
import time
import torch.nn as nn
import librosa
from tqdm.auto import tqdm
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
# from torch.utils.data.sampler import WeightedRandomSampler

import numpy as np
from tfrecord.torch.dataset import TFRecordDataset

import torch.nn.functional as F
from tfrecord import reader
from tfrecord import iterator_utils
from transformers.models.t5.modeling_t5 import T5Stack
from transformers.models.t5.configuration_t5 import T5Config
parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# parser.add_argument("--background-noise", type=str, default='datasets/speech_commands/train/_background_noise_', help='path of background noise')
parser.add_argument("--comment", type=str, default='', help='comment in tensorboard title')
parser.add_argument("--traindata_path", type=str, default='/data/traindatasets_small.tfrecord', help='')
parser.add_argument("--validdata_path", type=str, default='/data/validdatasets_small.tfrecord', help='')
parser.add_argument("--batch_size", type=int, default=128, help='batch size')
parser.add_argument("--dataload-workers-nums", type=int, default=4, help='number of workers for dataloader')
parser.add_argument("--weight-decay", type=float, default=1e-2, help='weight decay')
parser.add_argument("--optim", choices=['sgd', 'adam'], default='sgd', help='choices of optimization algorithms')
parser.add_argument("--learning-rate", type=float, default=5e-5, help='learning rate for optimization')
parser.add_argument("--lr-scheduler", choices=['plateau', 'step'], default='plateau', help='method to adjust learning rate')
parser.add_argument("--lr-scheduler-patience", type=int, default=5, help='lr scheduler plateau: Number of epochs with no improvement after which learning rate will be reduced')
parser.add_argument("--lr-scheduler-step-size", type=int, default=50, help='lr scheduler step: number of epochs of learning rate decay.')
parser.add_argument("--lr-scheduler-gamma", type=float, default=0.1, help='learning rate is multiplied by the gamma to decrease it')
parser.add_argument("--max-epochs", type=int, default=60, help='max number of epochs')
parser.add_argument("--resume", type=str, help='checkpoint file to resume')
parser.add_argument("--segwidth", type=int, default=64, help='')
parser.add_argument("--dropout", type=float, default=0.1, help='dropout rate')
parser.add_argument("--alpha", type=float, default=0.5, help='')
parser.add_argument("--beta", type=float, default=0.8, help='')
parser.add_argument("--gamma", type=float, default=0.2, help='')
parser.add_argument("--train_nums", type=int, default=429405, help='')
parser.add_argument("--valid_nums", type=int, default=72055, help='')
# parser.add_argument("--train_nums", type=int, default=64, help='')
# parser.add_argument("--valid_nums", type=int, default=64, help='')
parser.add_argument("--nfft", type=int, default=512, help='')
parser.add_argument("--dmodel", type=int, default=512, help='')
parser.add_argument("--layers", type=int, default=6, help='')
parser.add_argument("--d_layers", type=int, default=6, help='')
parser.add_argument("--usetrans", type=int, default=1, help='')
parser.add_argument("--usemaxpool", type=int, default=1, help='')
parser.add_argument("--features", type=str, default='melspec', help='')
args = parser.parse_args()

class MyConfig(T5Config):
    def __init__(self,
                 use_dense=False,
                 use_position_embed=True,
                 input_length=None,
                 **kwargs
                 ):
        self.use_dense = use_dense
        self.use_position_embed = use_position_embed
        self.input_length = input_length
        super().__init__(
            **kwargs)

config = MyConfig(vocab_size=91+args.segwidth, input_length=args.segwidth, use_position_embed=True, return_dict=False,use_dense=False, d_model=args.dmodel, d_kv=64, d_ff=1024, num_layers=args.layers, num_decoder_layers=None, num_heads=8, relative_attention_num_buckets=32, dropout_rate=args.dropout,
                  layer_norm_epsilon=1e-06, initializer_factor=1.0, feed_forward_proj='relu', is_encoder_decoder=True, use_cache=False,
                  bos_token_id=1,
                  pad_token_id=0,
                   eos_token_id=2, 
                decoder_start_token_id=1)

class CustomTFRecordDataset(TFRecordDataset):
    def __init__(self, data_path,
                 index_path,
                 description=None,
                 shuffle_queue_size=None,
                 transform=None,
                 sequence_description=None,
                 compression_type=None,
                 length=None,
                 ):
        super(CustomTFRecordDataset, self).__init__(data_path,
                 index_path,
                 description,
                 shuffle_queue_size,
                 transform,
                 sequence_description,
                 compression_type)
        # self.length=length
        self.max_samples = length
        self.index = 0

    def __iter__(self):
        while self.index < self.max_samples:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                shard = worker_info.id, worker_info.num_workers
                np.random.seed(worker_info.seed % np.iinfo(np.uint32).max)
            else:
                shard = None
            it = reader.tfrecord_loader(data_path=self.data_path,
                                        index_path=self.index_path,
                                        description=self.description,
                                        shard=shard,
                                        sequence_description=self.sequence_description,
                                        compression_type=self.compression_type)
            if self.shuffle_queue_size:
                it = iterator_utils.shuffle_iterator(it, self.shuffle_queue_size)
            if self.transform:
                it = map(self.transform, it)
            self.index += 1
            return it

            
    
class LearnableAbsolutePositionEmbedding(nn.Module):
    def __init__(self, max_position_embeddings, hidden_size):
        super().__init__()
        self.is_absolute = True
        self.embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.register_buffer('position_ids', torch.arange(max_position_embeddings))

    def forward(self, x):
        """
        return (b l d) / (b h l d)
        """
        position_ids = self.position_ids[:x.size(-2)]

        if x.dim() == 3:
            return x + self.embeddings(position_ids)[None, :, :]

        elif x.dim() == 4:
            h = x.size(1)
            x = rearrange(x, 'b h l d -> b l (h d)')
            x = x + self.embeddings(position_ids)[None, :, :]
            x = rearrange(x, 'b l (h d) -> b h l d', h=h)
            return x


class Thoegaze(nn.Module):
    def __init__(self,
        d_model=512, 
        d_layers=args.d_layers, 
        unet=False,
        use_transcription_loss=True,
        use_max_pooling=False
    ):
        super().__init__()
        self.d_model  = d_model
        self.unet = unet
        if args.features=='melspec':
            n_features=229
        else: n_features=int(args.nfft/2+1)  
        self.use_transcription_loss=use_transcription_loss
        encoder_config = copy.deepcopy(config)
        self.embedding = nn.Linear(n_features,self.d_model)
        self.s_encoder = T5Stack(encoder_config)
        self.t_encoder = T5Stack(encoder_config)
        self.decoder=T5Stack(encoder_config)
        s_decoder_config = copy.deepcopy(config)
        s_decoder_config.is_decoder = True
        s_decoder_config.is_encoder_decoder = False   
        s_decoder_config.input_length=int(args.segwidth*1.5) 
        s_decoder_config.num_layers = d_layers
      
        self.use_max_pooling=use_max_pooling
        if self.use_max_pooling:
           self.max_pool=nn.MaxPool2d((args.segwidth,1)) 
        self.linear = nn.Linear(self.d_model, 91+args.segwidth)
        # self.m = nn.Softmax(dim=-1)
        # self.relu = nn.ReLU()
        self.out =nn.Linear(self.d_model,n_features)

        self.tgt_emb = nn.Embedding(args.segwidth+91, d_model)
        self.pos_emb = LearnableAbsolutePositionEmbedding(args.segwidth, d_model
            )
        self.s_decoder= T5Stack(s_decoder_config,embed_tokens=self.tgt_emb)

    def forward(self, content=None,timbre=None,decoder_inputs=None, state=None,type='syth'):


        if decoder_inputs:
            decoder_inputs=[self._shift_right(di) for di in decoder_inputs]
        # transcription_out=[]

        if type=='syth':
            _t=timbre
            content=self.embedding(content)

            content = self.pos_emb(content)
            _s= self.s_encoder(inputs_embeds=content)[0]
        # x=self.relu(x)
        else:

            timbre=self.embedding(timbre)
            timbre = self.pos_emb(timbre)
            _t = self.t_encoder(inputs_embeds=timbre)[0]

            return _t







        y0= self.decoder(inputs_embeds=_s+_t)[0]



        y0=self.out(y0)


        return y0# required to return a state

    def _shift_right(self, input_ids):
        decoder_start_token_id = 1
        pad_token_id = 0

        assert (
            decoder_start_token_id is not None
        ), "self.model.config.decoder_start_token_id has to be defined. In T5 it is usually set to the pad_token_id. See T5 docs for more information"

        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
        shifted_input_ids[..., 0] = decoder_start_token_id
        
        assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."
        # replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

        assert torch.all(shifted_input_ids >= 0).item(), "Verify that `shifted_input_ids` has only positive values"

        return shifted_input_ids


sr = 25600 # Sample rate.
n_fft = args.nfft # fft points (samples)
frame_shift = int(sr/32) # seconds
frame_length = int(sr/32) # seconds
hop_length = int(sr*frame_shift) # samples.
win_length = n_fft # int(sr*frame_length) # samples.
# n_mels = 80 # Number of Mel banks to generate
power = 1.2 # Exponent for amplifying the predicted magnitude
n_iter = 100 # Number of inversion iterations
preemphasis = .97 # or None
max_db = 100
ref_db = 20
top_db = 15
def _mel_to_linear_matrix(sr, n_fft, n_mels):
    m = librosa.filters.mel(sr, n_fft, n_mels)
    m_t = np.transpose(m)
    p = np.matmul(m, m_t)
    d = [1.0 / x if np.abs(x) > 1.0e-8 else x for x in np.sum(p, axis=0)]
    return np.matmul(m_t, np.diag(d))

def melspectrogram2wav(mel):
    '''# Generate wave file from spectrogram'''
    # transpose
    mel = mel.T

    # de-noramlize
    mel = (np.clip(mel, 0, 1) * max_db) - max_db + ref_db

    # to amplitude
    mel = np.power(10.0, mel * 0.05)
    m = _mel_to_linear_matrix(sr, n_fft, 229)
    mag = np.dot(m, mel)

    # wav reconstruction
    wav = griffin_lim(mag)

    # de-preemphasis
    wav = signal.lfilter([1], [1, -preemphasis], wav)

    # trim
    wav, _ = librosa.effects.trim(wav)

    return wav.astype(np.float32)
def spectrogram2wav(mag):
    '''# Generate wave file from spectrogram'''
    # transpose
    mag = mag.T

    # de-noramlize
    mag = (np.clip(mag, 0, 1) * max_db) - max_db + ref_db

    # to amplitude
    mag = np.power(10.0, mag * 0.05)

    # wav reconstruction
    wav = griffin_lim(mag)

    # de-preemphasis
    wav = signal.lfilter([1], [1, -preemphasis], wav)

    # c
    wav, _ = librosa.effects.trim(wav)

    return wav.astype(np.float32)
def griffin_lim(spectrogram):
    '''Applies Griffin-Lim's raw.
    '''
    X_best = copy.deepcopy(spectrogram)
    for i in range(n_iter):
        X_t = invert_spectrogram(X_best)
        est = librosa.stft(X_t, n_fft, hop_length, win_length=win_length)
        phase = est / np.maximum(1e-8, np.abs(est))
        X_best = spectrogram * phase
    X_t = invert_spectrogram(X_best)
    y = np.real(X_t)

    return y

def invert_spectrogram(spectrogram):
    '''
    spectrogram: [f, t]
    '''
    return librosa.istft(spectrogram, hop_length, win_length=win_length, window="hann")


@torch.no_grad()
def predict(content,timbre):
    content,cpl=tokenize(audio=content,method='stft',sample_rate=25600,hop_width=int(25600/32),nfft=args.nfft,seg_width=args.segwidth,return_target=False)
    timbre,tpl=tokenize(audio=timbre,method='stft',sample_rate=25600,hop_width=int(25600/32),nfft=args.nfft,seg_width=args.segwidth,return_target=False)
    model.eval()

    if use_gpu:
        content = content.cuda()
        timbre = timbre.cuda()
        # forward/backward
    i=0
    _timbre=None
    while i< len(timbre):
        
        _t = model(timbre=timbre[i:i+args.batch_size,:,:],type='timbre')
        i+=args.batch_size
        _t=torch.reshape(_t,(-1,_t.size()[-1]))
        if args.usemaxpool:
            _t,_=torch.max(_t,0)
            if _timbre:
                _timbre=torch.stack(_t,_timbre)
                _timbre,_=torch.max(_timbre,0)
            else:
                _timbre=_t
        else:
            if _timbre:
                _timbre=torch.cat((_timbre,_t),0)
            else:
                _timbre=_t
    if not args.usemaxpool:
        _timbre=torch.mean(_timbre[:-tpl,:],0)   
    i=0
    spec=None
    while i< len(timbre):
        out = model(content=content[i:i+args.batch_size,:,:],timbre=_timbre)
        i+=args.batch_size
        out=torch.reshape(_t,(-1,out.size()[-1]))
        if spec:
            spec=torch.cat((spec,out),0)
        else:
            spec=out
    
    spec=spec[:-cpl,:]
    #2wav

    wav = melspectrogram2wav(spec)
    # librosa.output.write_wav("gg_stft.wav", wav, sr)
    librosa.output.write_wav("test.wav", wav, sr)


index_path = None

if __name__=="__main__":

    use_gpu = torch.cuda.is_available()
    print('use_gpu', use_gpu,'use_trans',args.usetrans)
    if use_gpu:
        torch.backends.cudnn.benchmark = True
    print('alpha',args.alpha,'beta',args.beta,'gamma',args.gamma)
    # model=Thoegaze(d_model=args.dmodel,use_transcription_loss=False,use_max_pooling=args.usemaxpool)
    if use_gpu:
        model=torch.load('thoagazer_s4_sgd_plateau_bs8_lr5.0e-05_wd1.0e-02_contrastive-best-los-tt.pth')
    else:
        model=torch.load('thoagazer_s4_sgd_plateau_bs8_lr5.0e-05_wd1.0e-02_contrastive-best-los-tt.pth',map_location=torch.device('cpu'))
    if isinstance(model,torch.nn.DataParallel):
        params = model.module
    # if use_gpu:
    # for k,v in params.items():
    #     print(398,k)
    model.eval()
    print(type(model))


    # predict()



