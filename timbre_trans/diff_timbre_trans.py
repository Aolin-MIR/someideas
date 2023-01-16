from unicodedata import bidirectional
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import gc
import os
from einops import rearrange 
import torch
import copy
import argparse
import time
import torch.nn as nn
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
from loss import STFTLoss
from audio_diffusion_pytorch import AudioDiffusionConditional
parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)


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
parser.add_argument("--maxepochs", type=int, default=30, help='max number of epochs')
parser.add_argument("--resume", type=str, help='checkpoint file to resume')
parser.add_argument("--segwidth", type=int, default=512, help='')
parser.add_argument("--dropout", type=float, default=0.1, help='dropout rate')
parser.add_argument("--alpha", type=float, default=1, help='')
parser.add_argument("--beta", type=float, default=1, help='')
parser.add_argument("--gamma", type=float, default=1, help='')
parser.add_argument("--train_nums", type=int, default=100000, help='')
parser.add_argument("--valid_nums", type=int, default=5000, help='')
parser.add_argument("--sr", type=int, default=16000, help='')
parser.add_argument("--nfft", type=int, default=2048, help='')
parser.add_argument("--dmodel", type=int, default=256, help='')
parser.add_argument("--layers", type=int, default=6, help='')
parser.add_argument("--d_layers", type=int, default=6, help='')
parser.add_argument("--usetrans", type=int, default=1, help='')
parser.add_argument("--pooling_type", type=str, default='gru', help='')
parser.add_argument("--nocross", type=int, default=0, help='')
parser.add_argument("--features", type=str, default='melspec', help='')
parser.add_argument("--vq", type=int, default=1, help='')
parser.add_argument("--auto_regrssion_decoder", type=int, default=1, help='')
args = parser.parse_args()
hopsize=80 if args.features=='melspec' else args.sr//32
winsize=200 if args.features=='melspec' else args.nfft

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
                  pad_token_id=0, eos_token_id=2, decoder_start_token_id=1)


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

class VQEmbedding(nn.Module):

    def __init__(self, use_codebook_loss=True, axis=-1):
        super().__init__()
        self.embedding = nn.Embedding(2048, args.dmodel)
        self._use_codebook_loss = use_codebook_loss
        # self._cfg['init'].bind(nn.init.kaiming_uniform_)(self.embedding.weight)
        self._axis = axis

    def forward(self, input):
        if self._axis != -1:
            input = input.transpose(self._axis, -1)

        distances = (torch.sum(input ** 2, axis=-1, keepdim=True)
                     - 2 * torch.matmul(input, self.embedding.weight.T)
                     + torch.sum(self.embedding.weight ** 2, axis=-1))
        ids = torch.argmin(distances, axis=-1)
        quantized = self.embedding(ids)

        losses = {
            'commitment': ((quantized.detach() - input) ** 2)
        }
        
        if self._use_codebook_loss:
            losses['codebook'] = ((quantized - input.detach()) ** 2)
            
            # Straight-through gradient estimator as in the VQ-VAE paper
            # No gradient for the codebook
            quantized = (quantized - input).detach() + input
        else:
            # Modified straight-through gradient estimator
            # The gradient of the result gets copied to both inputs (quantized and non-quantized)
            quantized = input + quantized - input.detach()

        if self._axis != -1:
            quantized = quantized.transpose(self._axis, -1).contiguous()

        return quantized, ids, losses            
    
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
        # pooling_type='gru'
    ):
        super().__init__()
        self.d_model  = d_model
        self.unet = unet
        self.use_transcription_loss=use_transcription_loss
        encoder_config = copy.deepcopy(config)
        if args.features=='melspec':
            n_features=80
        else: n_features=int(args.nfft/2+1)
        if args.vq:
            self.vq=VQEmbedding()
        self.embedding = nn.Linear(n_features,self.d_model)
        self.s_encoder = T5Stack(encoder_config)
        self.t_encoder = T5Stack(encoder_config)
        # decoder_config = copy.deepcopy(config)
        # decoder_config.is_decoder = True
        # decoder_config.is_encoder_decoder = False   
        # decoder_config.input_length=int(args.segwidth*1.5) 
        # decoder_config.num_layers = d_layers 
        if args.auto_regrssion_decoder:
            decoder_config = copy.deepcopy(config)
            decoder_config.is_decoder = True
            decoder_config.is_encoder_decoder = False   
            self.decoder=T5Stack(decoder_config)
  

        else:
            self.decoder=T5Stack(encoder_config)

        s_decoder_config = copy.deepcopy(config)
        s_decoder_config.is_decoder = True
        s_decoder_config.is_encoder_decoder = False   
        s_decoder_config.input_length=int(args.segwidth*1.5) 
        s_decoder_config.num_layers = d_layers
        self.pooling_type=args.pooling_type
        if self.pooling_type=='max':
           self.max_pool=nn.MaxPool2d((args.segwidth,1)) 
        elif self.pooling_type=='gru':
           self.gru=nn.GRU(self.d_model,self.d_model,2,bidirectional=True,dropout=args.dropout,batch_first=True) 
        self.linear = nn.Linear(self.d_model, 91+args.segwidth)
        # self.m = nn.Softmax(dim=-1)
        # self.relu = nn.ReLU()
        # self.dropout=nn.Dropout(p=args.dropout)
        self.out =nn.Linear(self.d_model, n_features)

        self.tgt_emb = nn.Embedding(args.segwidth+91, d_model)
        # self.pos_emb = LearnableAbsolutePositionEmbedding(args.segwidth, d_model)
        self.s_decoder= T5Stack(s_decoder_config,embed_tokens=self.tgt_emb)
        self.dif_decoder=AudioDiffusionConditional(
                in_channels=1,
                embedding_max_length=args.segwidth,
                embedding_features=self.d_model,
                embedding_mask_proba=args.dropout # Conditional dropout of batch elements
            )

    def forward(self, x_list,decoder_inputs=None, state=None):

        t=[]
        s=[]
        losses=[]
        d_x_list=[]
        if decoder_inputs:
            decoder_inputs=[self._shift_right(di) for di in decoder_inputs]
        transcription_out=[]
        for i, _x in enumerate(x_list):
            
            x=self.embedding(_x)

            # x = self.pos_emb(x)

            d_x_list.append(x)
            _s= self.s_encoder(inputs_embeds=x,return_dict=True).last_hidden_state
            if args.vq:
                _s,_,l=self.vq(_s)
                losses.append(l)
            _t = self.t_encoder(inputs_embeds=x,return_dict=True).last_hidden_state
            if self.pooling_type=='max':
                _t=self.max_pool(_t)
            elif self.pooling_type=='gru':
                _,ht=self.gru(_t)
                _t=ht[-1]
                _t=torch.unsqueeze(
                    _t,1)
            else:
                _t=torch.mean(_t,1)
                _t=torch.unsqueeze(
                    _t,1)

            t.append(_t)
            s.append(_s)


            if self.use_transcription_loss:
                if i%2==0:
                    decoder_input=decoder_inputs[0]
                else:
                    decoder_input=decoder_inputs[1]
                # decoder_input = self.tgt_emb(decoder_input)
                # decoder_input = self.pos_emb(decoder_input)
                # ds=decoder_input.size()[1]
                ss= _s.size()[1]
                bs=decoder_input.size()[0]
                mem_mask=None
                # if ss<ds:
                mem_mask=torch.ones(bs,int(ss))
                mem_mask=nn.functional.pad(mem_mask,(0,int(ss/2),0,0),'constant',value=0)  

                if use_gpu:
                    mem_mask=mem_mask.cuda()
                _s=nn.functional.pad(_s,(0,0,0,int(ss/2),0,0),'constant',value=0)  
                transcription_out.append(self.linear(self.s_decoder(input_ids=decoder_input,encoder_hidden_states=_s,encoder_attention_mask=mem_mask,use_cache=True,return_dict=False
                )[0]))

        if args.auto_regrssion_decoder:
            d_x_list=[nn.functional.pad(x[:,:-1,:],(0,0,1,0,0,0),'constant',value=0) for x in d_x_list]
            
            y0= self.decoder(inputs_embeds=d_x_list[0],encoder_hidden_states=s[0]+t[1],return_dict=True).last_hidden_state
            y1= self.decoder(inputs_embeds=d_x_list[1],encoder_hidden_states=s[1]+t[0],return_dict=True).last_hidden_state
            y2= self.decoder(inputs_embeds=d_x_list[2],encoder_hidden_states=s[2]+t[3],return_dict=True).last_hidden_state
            y3= self.decoder(inputs_embeds=d_x_list[3],encoder_hidden_states=s[3]+t[2],return_dict=True).last_hidden_state

            y0=self.out(y0)
            y1=self.out(y1)
            y2=self.out(y2)
            y3=self.out(y3)

            if not args.nocross:
                # y0= self.decoder(inputs_embeds=d_x_list[],encoder_hidden_states=s[2]+t[1])[0]
                # y1= self.decoder(inputs_embeds=d_x_list[],encoder_hidden_states=s[3]+t[0])[0]
                # y2= self.decoder(inputs_embeds=d_x_list[],encoder_hidden_states=s[0]+t[3])[0]
                # y3= self.decoder(inputs_embeds=d_x_list[],encoder_hidden_states=s[1]+t[2])[0]

                y0_= self.out(self.decoder(inputs_embeds=d_x_list[0],encoder_hidden_states=s[2]+t[1],return_dict=True).last_hidden_state)
                y1_= self.out(self.decoder(inputs_embeds=d_x_list[1],encoder_hidden_states=s[3]+t[0],return_dict=True).last_hidden_state)
                y2_= self.out(self.decoder(inputs_embeds=d_x_list[2],encoder_hidden_states=s[0]+t[3],return_dict=True).last_hidden_state)
                y3_= self.out(self.decoder(inputs_embeds=d_x_list[3],encoder_hidden_states=s[1]+t[2],return_dict=True).last_hidden_state)

        else:#DIFFUSION
                        
            y0= self.dif_decoder(x_list[0],embedding=s[0]+t[1])
            y1= self.dif_decoder(x_list[1],embedding=s[0]+t[1])
            y2= self.dif_decoder(x_list[2],embedding=s[0]+t[1])
            y3= self.dif_decoder(x_list[3],embedding=s[0]+t[1])

            if not args.nocross:
                # y0= self.decoder(inputs_embeds=s[2]+t[1])[0]
                # y1= self.decoder(inputs_embeds=s[3]+t[0])[0]
                # y2= self.decoder(inputs_embeds=s[0]+t[3])[0]
                # y3= self.decoder(inputs_embeds=s[1]+t[2])[0]

                y0_= self.out(self.dif_decoder(x_list[0],embedding=s[2]+t[1],return_dict=True).last_hidden_state)
                y1_= self.out(self.dif_decoder(x_list[1],embedding=s[3]+t[0],return_dict=True).last_hidden_state)
                y2_= self.out(self.dif_decoder(x_list[2],embedding=s[0]+t[3],return_dict=True).last_hidden_state)
                y3_= self.out(self.dif_decoder(x_list[3],embedding=s[1]+t[2],return_dict=True).last_hidden_state)

        

        out={}
        out['synout']=[y0,y1,y2,y3]
        if not args.nocross:
            out['crossout']=[y0_,y1_,y2_,y3_]

        
        if self.use_transcription_loss:

                out['trans']=transcription_out

        out['t']=t
        if args.vq:
            out['vq']=losses
        return out 
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


def criterion(outputs,inputs,score=None,alpha=0.5,beta=1,gamma=1,step=None):
    loss={}
    para={}
    
    # beta_s = 1
    # beta_s_anneal_start = 0
    # beta_s_anneal_steps = 0
    # if step is not None:
    #     if beta_s_anneal_steps == 0:
    #         beta_s = 0. if step < beta_s_anneal_start else beta_s
    #     else:
    #         beta_s *= min(1., max(0., (step - beta_s_anneal_start) / beta_s_anneal_steps))
    if args.vq:
        ls=outputs['vq']
        # outputs=outputs[:-4]
        para['commitment'] = 0.5
        para['codebook'] = 1
        loss['commitment']=sum([x['commitment'] for x in ls]).mean()
        loss['codebook']=sum([x['codebook'] for x in ls]).mean()
       
    if score:
        _s1,_s2,_s3,_s4=outputs['trans']
        # if args.nocross:
        #     y1,y2,y3,y4,_s1,_s2,_s3,_s4,t1,t2,t3,t4=outputs
        # else:
        #     y1,y2,y3,y4,y1_,y2_,y3_,y4_,_s1,_s2,_s3,_s4,t1,t2,t3,t4=outputs
        sa,sb=score
        if use_gpu:
            sa = sa.cuda().long()
            sb = sb.cuda().long()
        # sa=F.one_hot(sa,args.segwidth+91).float()
        # sb=F.one_hot(sb,args.segwidth+91).float()
        # print(271,sa.size())
        sfn = torch.nn.CrossEntropyLoss(size_average=True,reduce=True,ignore_index=0)

        para['trans']=gamma
        loss['trans'] = sfn(torch.transpose(_s1, -2, -1),sa) + sfn(torch.transpose(_s3, -2, -1),sa) +sfn(torch.transpose(_s2, -2, -1),sb)+ sfn(torch.transpose(_s4, -2, -1),sb)

    
        
    y1,y2,y3,y4=outputs['synout']

        
        

    x1,x2,x3,x4=inputs
    

    para['syth']=alpha
    if args.auto_regrssion_decoder==1:
        fn=STFTLoss(fft_size=args.nfft,hop_size=hopsize,win_size=winsize)
        loss['syth']=fn(x1,y1)+fn(x2,y2)+fn(x3,y3)+fn(x4,y4)
    else:
        loss['syth']=y1+y2+y3+y4
    if not args.nocross:
        y1_,y2_,y3_,y4_=outputs['crossout']
        triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)
        para['const']=beta
        loss['const']= triplet_loss(y1_,y1,y2)+triplet_loss(y2_,y2,y1)+triplet_loss(y3_,y3,y4)+triplet_loss(y4_,y4,y3)+triplet_loss(y1_,y1,y3)+triplet_loss(y2_,y2,y4)+triplet_loss(y3_,y3,y1)+triplet_loss(y4_,y4,y1)
        # loss_syth+=loss_t

    loss['total'] = sum(para[name]*loss for name, loss in loss.items()
                      )

    return loss

def note_f1_v3(outputs,sa,sb):
    '''Calculate note-F1 score.  
    Returns
    -------
    dict    
    '''
    # print(24, evalPrediction.predictions.shape)
    total_pred = 0
    total_true = 0
    count = 0
    _s1,_s2,_s3,_s4=outputs['trans']
    pred=torch.cat([_s1,_s2,_s3,_s4],axis=0)
    target=torch.cat([sa,sb,sa,sb],axis=0)

    for y_pred, y_true in zip(pred,target):

        assert y_true.ndim == 1
        assert y_pred.ndim == 1 or y_pred.ndim == 2

        if y_pred.ndim == 2:

            y_pred = y_pred.argmax(dim=1)

        y_pred = y_pred.tolist()

        try:
            y_pred = y_pred[:y_pred.index(0)]
        except ValueError as e:
            pass
        temp_t=[]
        segnum=-1
        for y in y_true:
            if y==2:
                break
            elif 3<=y<args.segwidth+3:
                segnum=y-3
            elif y>=args.segwidth+3:
                if segnum>=0:
                    temp_t.append((segnum,y-args.segwidth-3))
        temp_p=[]       
        segnum=-1
        for y in y_pred:
            if y==2:
                break
            elif 3<=y<args.segwidth+3:
                segnum=y-3
            elif y>=args.segwidth+3:
                if segnum>=0:
                    temp_p.append((segnum,y-args.segwidth-3))

        # temp_p = [((x)//88, (x) % 88)
        #         for x in temp_p ]

        total_pred += len(temp_p)
        # temp_t = temp_t.tolist()
        # temp_t = [x for x in temp_t]
        total_true += len(temp_t)
        for i in temp_t:

            relt_true = i[0]
            note_true = i[1]
            for j in temp_p[:]:
                if j[1] == note_true and j[0] in range(max(0, relt_true-5), relt_true+5):
                    count += 1
                    temp_p.remove(j)
                    break
        del temp_t
        del temp_p

    return {'count':count,'tt':total_true,'tp':total_pred}

def train(epoch):

    global global_step

    print("epoch %3d with lr=%.02e" % (epoch, get_lr()))
    phase = 'train'
    writer.add_scalar('%s/learning_rate' % phase,  get_lr(), epoch)

    model.train()  # Set model to training mode

    running_loss,running_loss_t,running_loss_codebook,running_loss_trans,running_loss_syth,running_loss_commitment = 0.0,0.0, 0.0, 0.0, 0.0,0.0
    it = 0
    total_it=args.train_nums//train_dataloader.batch_size

    # pbar = tqdm(train_dataloader, unit="audios", unit_scale=train_dataloader.batch_size,total=total_it)
    # print( 245,len(pbar))
    for batch in train_dataloader:
        if it==total_it:
            break

        x0,x1,x2,x3=batch['x0'],batch['x1'],batch['x2'],batch['x3']
        if args.usetrans:
            t0,t1=batch['t0'],batch['t1']
        del batch

        if use_gpu:
            x0 = x0.cuda()
            x1 = x1.cuda()
            x2 = x2.cuda()
            x3 = x3.cuda()
            if args.usetrans:
                t0 = t0.cuda()
                t1 = t1.cuda()




        # forward/backward
        if args.usetrans:

            outputs = model([x0,x1,x2,x3],[t0,t1])
            loss= criterion(outputs, [x0,x1,x2,x3],[t0,t1],alpha=args.alpha,beta=args.beta,gamma=args.gamma)
        else:

            outputs = model([x0,x1,x2,x3])
            loss= criterion(outputs, [x0,x1,x2,x3],None,alpha=args.alpha,beta=args.beta,gamma=args.gamma)
        optimizer.zero_grad()

        loss['total'].backward()
        optimizer.step()

        # statistics
        it += 1
        global_step += 1
        running_loss += loss['total'].item()
        running_loss_t += loss['const'].item()
        if args.vq:
            running_loss_commitment += loss['commitment'].item()
            running_loss_codebook += loss['codebook'].item()
            
        if args.usetrans:
            running_loss_trans += loss['trans'].item()

        running_loss_syth += loss['syth'].item()
        if it%20==0:
            print({'epoch':str(epoch+1),
                'train_loss': "%.05f" % (running_loss / it),
                'loss_commitment': "%.05f" % (running_loss_commitment / it),
                                'loss_codebook': "%.05f" % (running_loss_codebook / it),
                'loss_const': "%.05f" % (running_loss_t / it),
                'loss_trans': "%.05f" % (running_loss_trans / it),
                'loss_syth': "%.05f" % (running_loss_syth / it),
                
                        # '系统总计内存':"%.05f" % zj,
    #    '系统已经使用内存':"%.05f" % ysy,
    #     '系统空闲内存':"%.05f" % kx,
    #             '本进程占用内存(MB)':"%.05f" % (bj),
        })

        # print('本进程占用内存(MB)%.05f' % (bj))
    # accuracy = correct/total
    # epoch_loss = running_loss / it
    # print('epoch:',epoch+1,' done')
    # writer.add_scalar('%s/accuracy' % phase, 100*accuracy, epoch)
    # writer.add_scalar('%s/epoch_loss' % phase, epoch_loss, epoch)
@torch.no_grad()
def valid(epoch,es=0):
    global best_f1, best_loss, global_step,best_const_loss
    epsilon = 1e-7
    phase = 'valid'
    model.eval()  # Set model to evaluate mode
    print('start evaluating')
    running_loss,running_loss_t,running_loss_s,running_loss_trans,running_loss_syth = 0.0, 0.0, 0.0, 0.0, 0.0
    count,tt,tp=0,0,0
    best_f1=0.0
    it = 0
    nf1=-1
    total_it=args.valid_nums//valid_dataloader.batch_size
    # correct = 0
    # total = 0
    # adj=torch.zeros(88,88,dtype=torch.float32)
    # for j in range(88):
    #     for i in range(j%12,88,12):
    #         adj[j,i]=1
    # if use_gpu:
    #     adj=adj.cuda()
    # pbar = tqdm(valid_dataloader,
    #      unit="audios", unit_scale=valid_dataloader.batch_size,total=total_it)
    for batch in valid_dataloader:
        if it==total_it:
            break
        x0,x1,x2,x3=batch['x0'],batch['x1'],batch['x2'],batch['x3']
        if args.usetrans:
            t0,t1=batch['t0'],batch['t1']
        del batch

        if use_gpu:
            x0 = x0.cuda()
            x1 = x1.cuda()
            x2 = x2.cuda()
            x3 = x3.cuda()
            if args.usetrans:
                t0 = t0.cuda()
                t1 = t1.cuda()


        # forward/backward
        if args.usetrans:
            outputs = model([x0,x1,x2,x3],[t0,t1])
            loss = criterion(outputs, [x0,x1,x2,x3],[t0,t1],alpha=args.alpha,beta=args.beta,gamma=args.gamma)
        else:
            outputs = model([x0,x1,x2,x3])
            loss= criterion(outputs, [x0,x1,x2,x3],None,alpha=args.alpha,beta=args.beta,gamma=args.gamma)
        it += 1
        global_step += 1
        running_loss += loss['total'].item()
        # running_loss_s += loss_s
        running_loss_t += loss['const'].item()
        
        running_loss_syth += loss['syth'].item()
    #加速noteF1的计算
        if args.usetrans:
            running_loss_trans += loss['trans'].item()
            metric=note_f1_v3(outputs,t0,t1)
            # statistics

            count+=metric['count']
            tt+=metric['tt']
            tp+=metric['tp']
            r = count/(tt+epsilon)
            p = count/(tp+epsilon)

            nf1 = 2 * (p*r) / (r + p + epsilon)

        if it%5==0:
            print({'v_epoch':str(epoch+1),
                'train_loss': "%.05f" % (running_loss / it),

                'loss_const': "%.05f" % (running_loss_t / it),
                'loss_trans': "%.05f" % (running_loss_trans / it),
                'loss_syth': "%.05f" % (running_loss_syth / it),})
    # accuracy = correct/total
    epoch_loss = (running_loss_syth / it)#.tolist()
    epoch_const_loss=(running_loss_t / it)
    # writer.add_scalar('%s/accuracy' % phase, 100*accuracy, epoch)
    # writer.add_scalar('%s/epoch_loss' % phase, epoch_loss, epoch)

    checkpoint = {
        'epoch': epoch,
        'step': global_step,
        'state_dict': model.state_dict(),
        'loss': epoch_loss,
        'f1': nf1,
        'optimizer' : optimizer.state_dict(),
    }
    if args.usetrans:

        if  nf1> best_f1:
            best_f1 = nf1

    #     torch.save(checkpoint, 'checkpoints/best-loss-speech-commands-checkpoint-%s.pth' % full_name)
    #     torch.save(model, '%d-%s-best-loss.pth' % (start_timestamp, full_name))
    # print(498,epoch_loss,type(epoch_loss),best_loss,type(best_loss))
    print (checkpoint['loss'],checkpoint['f1'])
    if epoch_loss < best_loss:
        
        best_loss = epoch_loss
        torch.save(checkpoint, './checkpoints/best-loss-tt-checkpoint-%s.pth' % full_name)
        torch.save(model, './checkpoints/%s-best-los-tt.pth' % ( full_name))
        es=0
    else:
        es+=1
    del checkpoint  # reduce memory

    return epoch_loss
def get_lr():
    return optimizer.param_groups[0]['lr']

def parse_fn(features):

   
    features['x0'] =np.frombuffer(
        features['x0'], dtype=np.float32).reshape(args.segwidth, -1)
    features['x1'] =np.frombuffer(
        features['x1'], dtype=np.float32).reshape(args.segwidth, -1)
    features['x2'] =np.frombuffer(
        features['x2'], dtype=np.float32).reshape(args.segwidth, -1)
    features['x3'] =np.frombuffer(
        features['x3'], dtype=np.float32).reshape(args.segwidth, -1)
    # if features['inputs_embeds'].shape[-1] not in [512, nbins]:
    #     print(features['inputs_embeds'].shape, features['labels'])
    #     assert 1==2
    if args.usetrans:
        _features=eval(features['t0'])

        features['t0']=[1]
        for i,x in enumerate(_features):
            if not x==[]:
                features['t0'].append(i+3)
                for y in x:
                    features['t0'].append(y+3+args.segwidth)
        del _features
        _features1=eval(features['t1'])

        features['t1']=[1]
        for i,x in enumerate(_features1):

            if not x==[]:
                features['t1'].append(i+3)
                for y in x:
                    features['t1'].append(y+3+args.segwidth)
        del _features1
        if len(features['t0'])>int(1.5*args.segwidth-1):
            features['t0']=features['t0'][:int(1.5*args.segwidth)-1]+[2]
        else:
            features['t0'] += [2]#eos
            features['t0'] += [0]*(int(1.5*args.segwidth)-len(features['t0']))

        if len(features['t1'])>int(1.5*args.segwidth-1):
            features['t1']=features['t1'][:int(1.5*args.segwidth)-1]+[2]
        else:
            features['t1'] += [2]#eos
            features['t1'] += [0]*(int(1.5*args.segwidth)-len(features['t1']))
        # print(592,len(features['t1']))
        features['t0']=torch.tensor(features['t0'])
        features['t1']=torch.tensor(features['t1'])
    else:
        del features['t0']
        del features['t1']
    return features

# def predict(content,timbre):
    




index_path = None




if __name__=="__main__":


    use_gpu = torch.cuda.is_available()
    print('use_gpu', use_gpu,'use_trans',args.usetrans)
    if use_gpu:
        torch.backends.cudnn.benchmark = True
    print('alpha',args.alpha,'beta',args.beta,'gamma',args.gamma)

    train_path=args.traindata_path
    valid_path=args.validdata_path
    description = {"x0": "byte", "x1":"byte","x2": "byte", "x3":"byte","t0": "byte", "t1":"byte"}
    train_dataset = CustomTFRecordDataset(train_path, None, description,
                                    shuffle_queue_size=256, transform=parse_fn,length=args.train_nums)
    valid_dataset = CustomTFRecordDataset(valid_path, None, description,
                            transform=parse_fn,length=args.valid_nums)


    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,shuffle=False,
                                pin_memory=use_gpu, num_workers=args.dataload_workers_nums,)
                                # collate_fn=PadCollate())
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False,
                                pin_memory=use_gpu, num_workers=args.dataload_workers_nums,)
                                # collate_fn=PadCollate())


    # a name used to save checkpoints etc.
    full_name = '%s_%s_%s_bs%d_lr%.1e_wd%.1e' % ('dif', args.optim, args.lr_scheduler, args.batch_size, args.learning_rate, args.weight_decay)
    if args.comment:
        full_name = '%s_%s' % (full_name, args.comment)

    model = Thoegaze(d_model=args.dmodel,use_transcription_loss=args.usetrans)

    writer = SummaryWriter(comment=('double_trans' + full_name))
    if use_gpu:
        model = torch.nn.DataParallel(model).cuda()

    

    if args.optim == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    start_timestamp = int(time.time()*1000)
    start_epoch = 0
    best_accuracy = 0
    best_loss = 1e100
    best_const_loss = 1e100
    global_step = 0

    if args.resume:
        print("resuming a checkpoint '%s'" % args.resume)
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['state_dict'])
        model.float()
        optimizer.load_state_dict(checkpoint['optimizer'])

        # best_accuracy = checkpoint.get('accuracy', best_accuracy)
        best_loss = checkpoint.get('loss', best_loss)
        start_epoch = checkpoint.get('epoch', start_epoch)
        global_step = checkpoint.get('step', global_step)

        del checkpoint  # reduce memory

    if args.lr_scheduler == 'plateau':
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=args.lr_scheduler_patience, factor=args.lr_scheduler_gamma)
    else:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_scheduler_step_size, gamma=args.lr_scheduler_gamma, last_epoch=start_epoch-1)

    es=0
    best_loss=1e100


    print("training %s for thoegazer..." % 'transformer ')
    since = time.time()
    for epoch in range(start_epoch, args.maxepochs):

        if args.lr_scheduler == 'step':
            lr_scheduler.step()
        
        train(epoch)
        if use_gpu:
            torch.cuda.empty_cache()
        epoch_loss,es = valid(epoch,es)
        if es>5:
            print("early stop in epoch ",epoch)
            break
        if use_gpu:
            torch.cuda.empty_cache()
        if args.lr_scheduler == 'plateau':
            lr_scheduler.step(metrics=epoch_loss)

        time_elapsed = time.time() - since
        time_str = 'total time elapsed: {:.0f}h {:.0f}m {:.0f}s '.format(time_elapsed // 3600, time_elapsed % 3600 // 60, time_elapsed % 60)
        print("%s, best loss %f" % (time_str, best_loss))
    print("finished")

