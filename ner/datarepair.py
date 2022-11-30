from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import numpy as np
import random
from transformers import pipeline
import os
tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")

model = AutoModelForSeq2SeqLM.from_pretrained("google/mt5-small")


def normalize(x, e=0.05):
    tem = x
    if max(tem) == 0:
        tem += e
    return tem/tem.sum()
def score(tokens_tensor):
    loss = model(tokens_tensor, labels=tokens_tensor)[0]
    return np.exp(loss.cpu().detach().numpy())


unmasker = pipeline('fill-mask', 
# model='Multilingual-MiniLM-L12-H384'
model='bert-base-multilingual-uncased'
    )
# unmasker("Hello I'm a [MASK] model.")
def fill_mask(sen,pos,unmasker):
    # old=list(sen)
    old.insert(pos,'[MASK]')
    old=' '.join(old)
    # print(25,old)
    new=unmasker(old)
    # print(26,new)
    while 1:
        i=random.randint(0,len(new)-1)
        if not new[i]['token_str'] in sen[max(0,pos-1):min(len(sen)-1,pos+1)]:
            break
    return new[i]['sequence'].replace(" ","")

def tag_update(tags,i,b):
    if  b==0:
        return tags
    tags.insert(i,'O')
    return tags

def cgmh(sentence,tags):
    #sentence has been tokenizered as list
    old_score = score(sentence)
    # old_score = score(tokenizer.encode(sentence, add_special_tokens=False, return_tensors="pt"))
    i=0
    for _ in range(100):

        pos=i%(len(sentence))
        # i=random.randint(0,len(sentence)-1)

        #insert
        _sentence=fill_mask(sentence,i,unmasker)
        candidats=[sentence,_sentence]
        tokens_tensor = tokenizer.encode( _sentence, add_special_tokens=False, return_tensors="pt")
        insert_score=score(tokens_tensor)
        scores=[old_score,insert_score]
        # b=np.argmax([score_old, prob_change_prob, prob_changeanother_prob*0.3, prob_add_prob*0.1, prob_delete_prob*0.001])
        b=np.argmax([old_score, insert_score*0.1])
        tags=tag_update(tags,i,b)
        sentence=candidats[b]

        old_score=scores[b]
        if b!=0:
            print(sentence)
        # print(score)
        i+=1
    return tokenizer.encode(sentence, add_special_tokens=False, return_tensors="pt"),tags
if __name__ == '__main__':
    cgmh('作用就是控制这两种情况执行代码的过程')
    Filelist = []
    path=''
    for home, dirs, files in os.walk(path):
        for filename in files:
            # 文件名列表，包含完整路径

            if 'conll' == filename[-5:] :
                Filelist.append(os.path.join(home, filename))
          
            # # 文件名列表，只包含文件名
            # Filelist.append( filename)
    for fi in Filelist:
        f=open(f,'r')
        w=open('new_'+f,'w')
        sen=[]
        tags=[]
        index=0
        for line in f.readlines():
            if line.strip()=='':
                w.write(line.strip()+'\n')
                if not sen==[]:
                    _sen,_tags=cgmh(sen,tags)
                    assert(len(_sen)==len(_tags))
                    for s,t in tuple(_sen,_tags):
                        w.write(s+' _ _ '+t+'\n')

                tags = []
                sen = []
                index=0
                continue
            elif line[0]=='#':
                w.write(line.strip()+'\n')
                # tags=[]
                # sen=[]

            else:
                sen.append(line.split(' ')[0])
                tags.append(line.split(' ')[-1]=='O')

                index+=1

        f.close()
        w.close()
