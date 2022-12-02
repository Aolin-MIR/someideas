from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import numpy as np
import random
from transformers import pipeline
import os
import copy
tokenizer = AutoTokenizer.from_pretrained("google/mt5-base")
print(7,tokenizer)
model = AutoModelForSeq2SeqLM.from_pretrained("google/mt5-base")


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
model='bert-large-multilingual-cased'
    )
# unmasker("Hello I'm a [MASK] model.")
def fill_mask(old,pos,unmasker):
    # old=list(sen)
    old.insert(pos,'[MASK]')

    _old=' '.join(old)
    # print(25,old)
    new=unmasker(_old)
    # print(26,new)
    i=0
    while 1:
        
        # print(35,new[i],type(new[i]),i,len(new),old)
        # print(36,new[i]['token_str'])
        if not new[i]['token_str'] in old[max(0,pos-1):min(len(old)-1,pos+1)]:
            break
        i+=1
        
    c1=new[i]['token_str'] 
    j=i+1
    while 1:
        
        # print(35,new[i],type(new[i]),i,len(new),old)
        # print(36,new[i]['token_str'])
        if not new[i]['token_str'] in old[max(0, pos-1):min(len(old)-1, pos+1)]:
            break
        j+=1
    c2 = new[j]['token_str']
    old[pos] = c1  # new[i]['sequence']
    old1=copy.copy(old)
    old1[pos]=c2
    return old,old1
def tag_update(tags,i,b):
    if  b==0:
        return tags
    tags.insert(i,'O')
    return tags

def cgmh(sentence,tags):
    #sentence has been tokenizered as list
    # old_score = score(sentence)
    old_score = score(tokenizer.encode(''.join(sentence), add_special_tokens=False, return_tensors="pt"))
    i=1
    print('\n'+' '.join(sentence),tags)
    for _ in range(100):

        pos=i%(len(sentence))
        # print(60,tags[pos])
        if pos==0:
            i+=1
            continue
        if not tags[pos].strip() == 'O':
            # print(63, tags[pos])
            i+=1
            continue
        if not tags[max(0,pos-1)].strip() == 'O':
            i+=1
            continue
        # i=random.randint(0,len(sentence)-1)
        # print(64)
        #insert
        temp=copy.copy(sentence)
        # print(70,len(temp))
        _sentence1,_sentence2=fill_mask(temp,pos,unmasker)
        # print(72,len(_sentence))
        candidats=[sentence,_sentence1,_sentence2]
        # print(' '.join(sentence),_sentence)
        tokens_tensor1 = tokenizer.encode( ''.join(_sentence1), add_special_tokens=False, return_tensors="pt")
        tokens_tensor2 = tokenizer.encode( ''.join(_sentence2), add_special_tokens=False, return_tensors="pt")
        insert_score1=score(tokens_tensor1)
        insert_score2 = score(tokens_tensor2)
        scores = [old_score, insert_score1, insert_score2]
        # print(scores)
        # b=np.argmax([score_old, prob_change_prob, prob_changeanother_prob*0.3, prob_add_prob*0.1, prob_delete_prob*0.001])
        b=np.argmax([old_score, insert_score1*0.001,insert_score2*0.001])
        
        tags=tag_update(tags,pos,b)
        
        sentence=candidats[b]
        # print(80, len(tags), len(sentence),b)
        old_score=scores[b]
        if b!=0:
            print(scores)
            print(' '.join(sentence))
        
        # sentence=sentence.split(' ')
        i+=1
    return sentence,tags
if __name__ == '__main__':
    # cgmh('作用就是控制这两种情况执行代码的过程')
    Filelist = []
    path='train_dev_ori'
    for home, dirs, files in os.walk(path):
        for filename in files:
            # 文件名列表，包含完整路径

            if 'conll' == filename[-5:] :#and 'zh' in filename:
                Filelist.append(os.path.join(home, filename))
          
            # # 文件名列表，只包含文件名
            # Filelist.append( filename)
    for fi in Filelist:
        f=open(fi,'r')
        w=open(fi+'new.conll','w')
        sen=[]
        tags=[]
        index=0
        for line in f.readlines():
            if line.strip()=='':
                w.write(line.strip())
                if not sen==[]:
                    # print(109)
                    assert(len(sen) == len(tags))
                    _sen,_tags=cgmh(sen,tags)
                    # except Exception as e:
                    #     print(e, sen)
                    #     os._exit()

                    assert(len(_sen)==len(_tags))
                    for s,t in zip(_sen,_tags):
                        try:
                            w.write(s+' _ _ '+t)
                        except Exception as e:
                            print(e,s,t)

                tags = []
                sen = []
                index=0
                continue
            elif line[0]=='#':
                w.write(line.strip())
                # tags=[]
                # sen=[]

            else:
                # print(132)
                sen.append(line.split(' ')[0].strip())
                tags.append(line.split(' ')[-1].strip())

                index+=1

        f.close()
        w.close()

a=[{'score': 0.1596946269273758, 'token': 131, 'token_str': ':', 'sequence': '[CLS] [MASK] : ritratto di ranuccio farnese 1542 olio su tela 90 × 74 cm washington national gallery of art [SEP]'}, {'score': 0.13447661697864532, 'token': 117, 'token_str': ',', 'sequence': '[CLS] [MASK], ritratto di ranuccio farnese 1542 olio su tela 90 × 74 cm washington national gallery of art [SEP]'}, {'score': 0.1100836768746376, 'token': 119, 'token_str': '.', 'sequence': '[CLS] [MASK]. ritratto di ranuccio farnese 1542 olio su tela 90 × 74 cm washington national gallery of art [SEP]'}, {'score': 0.0819583460688591, 'token': 114, 'token_str': ')', 'sequence': '[CLS] [MASK] ) ritratto di ranuccio farnese 1542 olio su tela 90 × 74 cm washington national gallery of art [SEP]'}, {'score': 0.057345177978277206, 'token': 10145, 'token_str': 'il', 'sequence': '[CLS] [MASK] il ritratto di ranuccio farnese 1542 olio su tela 90 × 74 cm washington national gallery of art [SEP]'}]
b= [[{'score': 0.307655394077301, 'token': 117, 'token_str': ',', 'sequence': '[CLS] ii, [MASK] ritratto di ranuccio farnese 1542 olio su tela 90 × 74 cm washington national gallery of art [SEP]'}, {'score': 0.17222651839256287, 'token': 118, 'token_str': '-', 'sequence': '[CLS] ii - [MASK] ritratto di ranuccio farnese 1542 olio su tela 90 × 74 cm washington national gallery of art [SEP]'}, {'score': 0.13017362356185913, 'token': 119, 'token_str': '.', 'sequence': '[CLS] ii. [MASK] ritratto di ranuccio farnese 1542 olio su tela 90 × 74 cm washington national gallery of art [SEP]'}, {'score': 0.02380351908504963, 'token': 131, 'token_str': ':', 'sequence': '[CLS] ii : [MASK] ritratto di ranuccio farnese 1542 olio su tela 90 × 74 cm washington national gallery of art [SEP]'}, {'score': 0.01778019778430462, 'token': 10424, 'token_str': 'ed', 'sequence': '[CLS] ii ed [MASK] ritratto di ranuccio farnese 1542 olio su tela 90 × 74 cm washington national gallery of art [SEP]'}], [{'score': 0.11171279847621918, 'token': 10145, 'token_str': 'il', 'sequence': '[CLS] ii [MASK] il ritratto di ranuccio farnese 1542 olio su tela 90 × 74 cm washington national gallery of art [SEP]'}, {'score': 0.05920857936143875, 'token': 119, 'token_str': '.', 'sequence': '[CLS] ii [MASK]. ritratto di ranuccio farnese 1542 olio su tela 90 × 74 cm washington national gallery of art [SEP]'}, {'score': 0.03591380640864372, 'token': 117, 'token_str': ',', 'sequence': '[CLS] ii [MASK], ritratto di ranuccio farnese 1542 olio su tela 90 × 74 cm washington national gallery of art [SEP]'}, {'score': 0.03550481051206589, 'token': 131, 'token_str': ':', 'sequence': '[CLS] ii [MASK] : ritratto di ranuccio farnese 1542 olio su tela 90 × 74 cm washington national gallery of art [SEP]'}, {'score': 0.028115209192037582, 'token': 10365, 'token_str': 'ii', 'sequence': '[CLS] ii [MASK] ii ritratto di ranuccio farnese 1542 olio su tela 90 × 74 cm washington national gallery of art [SEP]'}]]
