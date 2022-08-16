import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import torch
# from transformer import MultiHeadAttentionLayer
from layers import Encoder,PositionalEncoding
import argparse
import time
import torch.nn as nn
# from src.models.sequence.ss.standalone.s4 import LinearActivation, S4
# from einops import rearrange
from tqdm.auto import tqdm
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from torch.autograd import Variable
import numpy as np
from tfrecord.torch.dataset import TFRecordDataset
parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# parser.add_argument("--train-dataset", type=str, default='datasets/speech_commands/train', help='path of train dataset')
# parser.add_argument("--valid-dataset", type=str, default='datasets/speech_commands/valid', help='path of validation dataset')
# parser.add_argument("--background-noise", type=str, default='datasets/speech_commands/train/_background_noise_', help='path of background noise')
parser.add_argument("--comment", type=str, default='', help='comment in tensorboard title')
parser.add_argument("--batch_size", type=int, default=16, help='batch size')
parser.add_argument("--dataload-workers-nums", type=int, default=4, help='number of workers for dataloader')
parser.add_argument("--weight-decay", type=float, default=1e-2, help='weight decay')
parser.add_argument("--optim", choices=['sgd', 'adam'], default='sgd', help='choices of optimization algorithms')
parser.add_argument("--learning-rate", type=float, default=1e-5, help='learning rate for optimization')
parser.add_argument("--lr-scheduler", choices=['plateau', 'step'], default='plateau', help='method to adjust learning rate')
parser.add_argument("--lr-scheduler-patience", type=int, default=5, help='lr scheduler plateau: Number of epochs with no improvement after which learning rate will be reduced')
parser.add_argument("--lr-scheduler-step-size", type=int, default=50, help='lr scheduler step: number of epochs of learning rate decay.')
parser.add_argument("--lr-scheduler-gamma", type=float, default=0.1, help='learning rate is multiplied by the gamma to decrease it')
parser.add_argument("--max-epochs", type=int, default=70, help='max number of epochs')
parser.add_argument("--resume", type=str, help='checkpoint file to resume')
parser.add_argument("--segwidth", type=int, default=64, help='')
parser.add_argument("--dropout", type=float, default=0.1, help='dropout rate')
parser.add_argument("--alpha", type=float, default=0.5, help='')
parser.add_argument("--beta", type=float, default=0.8, help='')
parser.add_argument("--gamma", type=float, default=0.2, help='')
parser.add_argument("--train_nums", type=int, default=429405, help='')
parser.add_argument("--valid_nums", type=int, default=72055, help='')
parser.add_argument("--nfft", type=int, default=512, help='')
parser.add_argument("--dmodel", type=int, default=512, help='')
args = parser.parse_args()

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
        self.length=length
    def __len__(self):
        return self.length


class Thoegaze(nn.Module):
    def __init__(self,
        d_model=512, 
        n_layers=6, 
        # pool=[4, 4], 
        # expand=2, 
        # ff=2, 
        # bidirectional=False,
        # glu=True,
        unet=False,
        dropout=0.0,
        use_transcription_loss=True
    ):
        super().__init__()
        self.d_model = H = d_model
        self.unet = unet
        self.use_transcription_loss=use_transcription_loss
        self.embedding = nn.Linear(args.nfft/2+1,self.d_model)
        self.pos_emb = PositionalEncoding(
            d_model=d_model,
            dropout=dropout)
        self.encoder = Encoder(d_model,2048,64,64,8,n_layers,0)
        self.decoder= Encoder(d_model,2048,64,64,8,n_layers,0)
        # en_layers,de_layers=[],[]
        # for _ in range(n_layers):

        #     en_layers.append(MultiHeadAttentionLayer(self.d_model*4,8,dropout))


        # self.encoder = nn.ModuleList(en_layers)
        # for _ in range(n_layers):

        #     de_layers.append(MultiHeadAttentionLayer(self.d_model*4,8,dropout))

        
        # self.decoder = nn.ModuleList(de_layers)

        # self.norm = nn.LayerNorm(H)
        self.linear = nn.Linear(int(self.d_model/2), 88)
        self.m = nn.Sigmoid()
        self.out =nn.Linear(self.d_model, args.nfft/2+1)
        assert H == d_model
   
    def forward(self, x_list, state=None):
        """

        """


        t=[]
        s=[]

        transcription_out=[]
        for x in x_list:
            # x=x.transpose(1, 2)
            x=self.embedding(x)

            x = self.pos_emb(x)

            x,_ = self.encoder(x)
            bs,l,d=x.shape
            x=x.view(bs,l,-1,2)
            _s=x[:,:,:,0]
            _t=x[:,:,:,1]
            t.append(_t)
            s.append(_s)
            #Todo:transcription decoder
            if self.use_transcription_loss:
                transcription_out.append(self.m(self.linear(_s)))



        y0,_= self.decoder(torch.cat((s[2],t[1]),axis=-1))
        y1,_= self.decoder(torch.cat((s[3],t[0]),axis=-1))
        y2,_= self.decoder(torch.cat((s[0],t[3]),axis=-1))
        y3,_= self.decoder(torch.cat((s[1],t[2]),axis=-1))

        y0=self.out(y0)
        y1=self.out(y1)
        y2=self.out(y2)
        y3=self.out(y3)

        if self.use_transcription_loss:
            return [y0,y1,y2,y3]+transcription_out
        else:
            return [y0,y1,y2,y3] # required to return a state




def criterion(outputs,inputs,score=None,alpha=0.5,beta=1,gamma=1):
    
    if score:
        y1,y2,y3,y4,_s1,_s2,_s3,_s4=outputs
        sa,sb=score
        sa = sa.float().cuda()
        sb = sb.float().cuda()
        pos_weight = torch.ones([88]).cuda()
        bcefn=nn.BCELoss(weight=pos_weight)
        # print(184,_s1.device,sa.device)
        loss_transcription = bcefn(_s1,sa) + bcefn(_s3,sa) +bcefn(_s2,sb)+ bcefn(_s4,sb)
    else:
        y1,y2,y3,y4=outputs

    x1,x2,x3,x4=inputs
    fn=torch.nn.MSELoss()
    # loss_s=fn(s1,s3)+fn(s2,s4) 
    # loss_t=fn(t1,t2)+fn(t3,t4)
    loss_syth=fn(x1,y1)+fn(x2,y2)+fn(x3,y3)+fn(x4,y4)
    if score:
        return beta*loss_syth+gamma*loss_transcription,loss_transcription,loss_syth
    else:
        return  beta*loss_syth,loss_syth
    
def note_f1(outputs,sa,sb):

    '''Calculate note-F1 score.  
    Returns
    -------
    dict    
    '''
    # print(24, evalPrediction.predictions.shape)
    total_pred=0
    total_true=0
    count =0
    _s1,_s2,_s3,_s4=outputs[-4:]
    def temp(pred,target):
        c=0
        tp=0
        pred=torch.round(pred)
        for x,y in zip(pred,target):
        
            for i,seg in enumerate(x):#
                for j,sheet in enumerate(seg):
                    # sheet = int(sheet)
                    if not sheet==0:
                        tp+=1
                        for z in range(max(0,i-5),min(args.segwidth,i+5)):
                            sc=j%12
                            while sc<88:
                                if y[z][sc]==1:
                                    c+=1
                                    break
                                sc+=12
                            
        tt=torch.count_nonzero(target)
        return tp,tt,c
            
    tp,tt,c=temp(_s1,sa)
    total_pred+=tp
    total_true+=tt
    count+=c
    tp,tt,c=temp(_s3,sa)
    total_pred+=tp
    total_true+=tt
    count+=c            
    tp,tt,c=temp(_s2,sb)
    total_pred+=tp
    total_true+=tt
    count+=c
    tp,tt,c=temp(_s4,sb)
    total_pred+=tp
    total_true+=tt
    count+=c

    epsilon = 1e-7
    # print(57,total_true,total_pred)
    r = count/(total_true+epsilon)
    p = count/(total_pred+epsilon)

    f1 = 2 * (p*r) / (r + p + epsilon)
    # print({'note_f1': f1, 'precision': p, 'recall': r})
    return {'note_f1':f1,'precision':p,'recall':r,'count':count,'tt':total_true,'tp':total_pred}

def train(epoch):
    global global_step

    print("epoch %3d with lr=%.02e" % (epoch, get_lr()))
    phase = 'train'
    writer.add_scalar('%s/learning_rate' % phase,  get_lr(), epoch)

    model.train()  # Set model to training mode

    running_loss,running_loss_t,running_loss_s,running_loss_trans,running_loss_syth = 0.0, 0.0, 0.0, 0.0,0.0
    it = 0
    total_it=args.train_nums//train_dataloader.batch_size

    pbar = tqdm(train_dataloader, unit="audios", unit_scale=train_dataloader.batch_size,total=total_it)
    # print( 245,len(pbar))
    for batch in pbar:
        if it==total_it:
            break
        # batch=next(iter(train_dataloader))
        # inputs = batch['input']
        # inputs = torch.unsqueeze(inputs, 1)
        # targets = batch['target']
        x0,x1,x2,x3,t0,t1=batch['x0'],batch['x1'],batch['x2'],batch['x3'],batch['t0'],batch['t1']

        if use_gpu:
            x0 = x0.cuda()
            x1 = x1.cuda()
            x2 = x2.cuda()
            x3 = x3.cuda()
            t0 = t0.cuda()
            t1 = t1.cuda()



        # forward/backward
        outputs = model([x0,x1,x2,x3])
        
        loss,loss_trans,loss_syth = criterion(outputs, [x0,x1,x2,x3],[t0,t1],alpha=args.alpha,beta=args.beta,gamma=args.gamma)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # statistics
        it += 1
        global_step += 1
        running_loss += loss

        running_loss_trans += loss_trans
        running_loss_syth += loss_syth
        # pred = outputs.data.max(1, keepdim=True)[1]
        # correct += pred.eq(targets.data.view_as(pred)).sum()
        # total += targets.size(0)

        # writer.add_scalar('%s/loss' %  loss, 
        # '%s/loss_t' %  loss_t, 
        # '%s/loss_s' % loss_s, 
        # '%s/loss_trans' % loss_trans
        # )

        # update the progress bar
        pbar.set_postfix({'epoch':str(epoch+1),
            'train_loss': "%.05f" % (running_loss / it),
            # 'loss_s': "%.05f" % (running_loss_s / it),
            # 'loss_t': "%.05f" % (running_loss_t / it),
            'loss_trans': "%.05f" % (running_loss_trans / it),
            'loss_syth': "%.05f" % (running_loss_syth / it),
        })

    # accuracy = correct/total
    epoch_loss = running_loss / it
    print('epoch:',epoch+1,' done')
    # writer.add_scalar('%s/accuracy' % phase, 100*accuracy, epoch)
    # writer.add_scalar('%s/epoch_loss' % phase, epoch_loss, epoch)
@torch.no_grad()
def valid(epoch):
    global best_f1, best_loss, global_step
    epsilon = 1e-7
    phase = 'valid'
    model.eval()  # Set model to evaluate mode
    print('start evaluating')
    running_loss,running_loss_t,running_loss_s,running_loss_trans,running_loss_syth = 0.0, 0.0, 0.0, 0.0, 0.0
    count,tt,tp=0,0,0
    it = 0
    total_it=args.valid_nums//valid_dataloader.batch_size
    # correct = 0
    # total = 0

    pbar = tqdm(valid_dataloader,
         unit="audios", unit_scale=valid_dataloader.batch_size,total=total_it)
    for batch in pbar:
        if it==total_it:
            break

        x0,x1,x2,x3,t0,t1=batch['x0'],batch['x1'],batch['x2'],batch['x3'],batch['t0'],batch['t1']


        if use_gpu:
            x0 = x0.cuda()
            x1 = x1.cuda()
            x2 = x2.cuda()
            x3 = x3.cuda()
            t0 = t0.cuda()
            t1 = t1.cuda()



        # forward/backward
        outputs = model([x0,x1,x2,x3])
        
        loss,loss_trans,loss_syth  = criterion(outputs, [x0,x1,x2,x3],[t0,t1],alpha=args.alpha,beta=args.beta,gamma=args.gamma)
        metric=note_f1(outputs,t0,t1)
        # statistics
        it += 1
        global_step += 1
        running_loss += loss
        # running_loss_s += loss_s
        # running_loss_t += loss_t
        running_loss_trans += loss_trans
        running_loss_syth += loss_syth
        count+=metric['count']
        tt+=metric['tt']
        tp+=metric['tp']
        r = count/(tt+epsilon)
        p = count/(tp+epsilon)

        nf1 = 2 * (p*r) / (r + p + epsilon)
        # writer.add_scalar('%s/loss' %  loss, 
        # '%s/loss_t' %  loss_t, 
        # '%s/loss_s' % loss_s, 
        # '%s/loss_trans' % loss_trans,
        
        # '%s/note_f1' % metric['note_f1']

        # )

        # update the progress bar
        pbar.set_postfix({
            'valid_loss': "%.05f" % (running_loss / it),
            # 'loss_s': "%.05f" % (running_loss_s / it),
            # 'loss_t': "%.05f" % (running_loss_t / it),
            'loss_trans': "%.05f" % (running_loss_trans / it),
            'loss_syth': "%.05f" % (running_loss_syth / it),
            'note_f1':"%.05f" % (nf1)
        })

    # accuracy = correct/total
    epoch_loss = running_loss / it
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

    if metric['note_f1'] > best_f1:
        best_f1 = metric['note_f1']
    #     torch.save(checkpoint, 'checkpoints/best-loss-speech-commands-checkpoint-%s.pth' % full_name)
    #     torch.save(model, '%d-%s-best-loss.pth' % (start_timestamp, full_name))
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        torch.save(checkpoint, '/common-data/liaolin/timbre_trans/checkpoints/best-loss-tt-checkpoint-%s.pth' % full_name)
        torch.save(model, '/common-data/liaolin/timbre_trans/checkpoints/%d-%s-best-los-tt.pth' % (start_timestamp, full_name))

    torch.save(checkpoint, '/common-data/liaolin/timbre_trans/checkpoints/last-tt.pth')
    del checkpoint  # reduce memory

    return epoch_loss
def get_lr():
    return optimizer.param_groups[0]['lr']

def parse_fn(features):
    # print(60, features['inputs_embeds'].shape,type(features['inputs_embeds']))
   
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
    _features=eval(features['t0'])
    # print(707,_features)
    _temp=np.zeros((args.segwidth,88),int)
    for i,x in enumerate(_features):

        for y in x:
            _temp[i][y]=1
    features['t0']=_temp        
    _features=eval(features['t1'])
    # print(707,type(_features))
    _temp=np.zeros((args.segwidth,88),int)
    for i,x in enumerate(_features):

        for y in x:
            _temp[i][y]=1    
    features['t1']=_temp

    # features['x0'] = torch.from_numpy(features['x0'],copy=)
    # features['x1'] = torch.from_numpy(features['x1'])
    # features['x2'] = torch.from_numpy(features['x2'])
    # features['x3'] = torch.from_numpy(features['x3'])
    # features['t0'] = torch.from_numpy(features['t0'])
    # features['t1'] = torch.from_numpy(features['t1'])

    return features


index_path = None




if __name__=="__main__":


    use_gpu = torch.cuda.is_available()
    print('use_gpu', use_gpu)
    if use_gpu:
        torch.backends.cudnn.benchmark = True
    print('alpha',args.alpha,'beta',args.beta,'gamma',args.gamma)
    # n_mels = 32
    # if args.input == 'mel40':
    #     n_mels = 40

    # data_aug_transform = Compose([ChangeAmplitude(), ChangeSpeedAndPitchAudio(), FixAudioLength(), ToSTFT(), StretchAudioOnSTFT(), TimeshiftAudioOnSTFT(), FixSTFTDimension()])
    # bg_dataset = BackgroundNoiseDataset(args.background_noise, data_aug_transform)
    # add_bg_noise = AddBackgroundNoiseOnSTFT(bg_dataset)
    # train_feature_transform = Compose([ToMelSpectrogramFromSTFT(n_mels=n_mels), DeleteSTFT(), ToTensor('mel_spectrogram', 'input')])
    train_path='./mae_timbre_small0805.tfrecord'
    valid_path='./mae_timbre_small0805_valid.tfrecord'
    description = {"x0": "byte", "x1":"byte","x2": "byte", "x3":"byte","t0": "byte", "t1":"byte"}
    train_dataset = CustomTFRecordDataset(train_path, None, description,
                                    shuffle_queue_size=256, transform=parse_fn,length=args.train_nums)
    valid_dataset = CustomTFRecordDataset(valid_path, None, description,
                            transform=parse_fn,length=args.valid_nums)

    # weights = train_dataset.make_weights_for_balanced_classes()
    # sampler = WeightedRandomSampler(weights, len(weights))
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,shuffle=False,
                                pin_memory=use_gpu, num_workers=args.dataload_workers_nums)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False,
                                pin_memory=use_gpu, num_workers=args.dataload_workers_nums)


    # a name used to save checkpoints etc.
    full_name = '%s_%s_%s_bs%d_lr%.1e_wd%.1e' % ('thoagazer_s4', args.optim, args.lr_scheduler, args.batch_size, args.learning_rate, args.weight_decay)
    if args.comment:
        full_name = '%s_%s' % (full_name, args.comment)

    model = Thoegaze(dropout=args.dropout,d_model=args.dmodel)

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




    print("training %s for thoegazer..." % 'transformer ')
    since = time.time()
    for epoch in range(start_epoch, args.max_epochs):
        if args.lr_scheduler == 'step':
            lr_scheduler.step()
        
        train(epoch)
        torch.cuda.empty_cache()
        epoch_loss = valid(epoch)
        torch.cuda.empty_cache()
        if args.lr_scheduler == 'plateau':
            lr_scheduler.step(metrics=epoch_loss)

        time_elapsed = time.time() - since
        time_str = 'total time elapsed: {:.0f}h {:.0f}m {:.0f}s '.format(time_elapsed // 3600, time_elapsed % 3600 // 60, time_elapsed % 60)
        print("%s, best loss %f" % (time_str, best_loss))
    print("finished")
