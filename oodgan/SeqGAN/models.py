import numpy as np
# import keras
import keras.backend as K
from keras.models import Model
from keras.layers import Input, Lambda, Activation, Dropout, Concatenate,Bidirectional
from keras.layers import Dense, Embedding, LSTM, GRU,Conv1D, GlobalMaxPooling1D
from keras.layers.wrappers import TimeDistributed
from keras.utils import to_categorical
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
import tensorflow as tf
import pickle
# from tqdm import tqdm
import code



def GeneratorPretraining(V, E, H):
    '''
    Model for Generator pretraining. This model's weights should be shared with
        Generator.
    # Arguments:
        V: int, Vocabrary size
        E: int, Embedding size
        H: int, LSTM hidden size
    # Returns:
        generator_pretraining: keras Model
            input: word ids, shape = (B, T)
            output: word probability, shape = (B, T, V)
    '''
    # in comment, B means batch size, T means lengths of time steps.
    input = Input(shape=(None,), dtype='int32', name='Input') # (B, T)
    out = Embedding(V, E, mask_zero=True, name='Embedding')(input) # (B, T, E)
    out = LSTM(H, return_sequences=True, name='LSTM')(out)  # (B, T, H)
    out = TimeDistributed(
        Dense(V, activation='softmax', name='DenseSoftmax'),
        name='TimeDenseSoftmax')(out)    # (B, T, V)
    generator_pretraining = Model(input, out)
    return generator_pretraining

class Generator():
    'Create Generator, which generate a next word.'
    def __init__(self, sess, B, V, E, H, lr=1e-3,use_gpt=False,clip=1):
        '''
        # Arguments:
            B: int, Batch size
            V: int, Vocabrary size
            E: int, Embedding size
            H: int, LSTM hidden size
        # Optional Arguments:
            lr: float, learning rate, default is 0.001
            clip: float, gradient clip value
        '''
        self.sess = sess
        self.B = B
        self.V = V
        self.E = E
        self.H = H
        self.lr = lr
        self.clip=clip
        self._build_gragh()
        self.reset_rnn_state()


    def _build_gragh(self):
        state_in = tf.placeholder(tf.float32, shape=(None, 1))
        h_in = tf.placeholder(tf.float32, shape=(None, self.H))
        c_in = tf.placeholder(tf.float32, shape=(None, self.H))
        action = tf.placeholder(tf.float32, shape=(None, self.V)) # onehot (B, V)
        reward  =tf.placeholder(tf.float32, shape=(None, )) # (B, )每一个batch的reward
        self.topk = tf.placeholder(tf.int32,shape=(None))
        self.layers = []

        embedding = Embedding(self.V, self.E, mask_zero=True, name='Embedding')
        out = embedding(state_in) # (B, 1, E) 将state_in输入embedding
        self.layers.append(embedding)

        lstm = LSTM(self.H, return_state=True, name='LSTM')
        out, next_h, next_c = lstm(out, initial_state=[h_in, c_in])  # (B, H)
        self.layers.append(lstm)

        dense = Dense(self.V, activation='softmax', name='DenseSoftmax')
        prob = dense(out)    # (B, V)
        self.layers.append(dense)

        log_prob = tf.log(tf.reduce_mean(prob * action, axis=-1)) # (B, )
        loss = - log_prob * reward
        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        grads = optimizer.compute_gradients(loss)
        for i, (g, v) in enumerate(grads):
            if g is not None:
                grads[i] = (tf.clip_by_norm(g, self.clip), v)  
        minimize = optimizer.apply_gradients(grads)

        self.state_in = state_in
        self.h_in = h_in
        self.c_in = c_in
        self.action = action
        self.reward = reward
        self.prob = prob
        self.temp=tf.nn.top_k(self.prob,k=self.topk)
        # self.probtopk=self.temp[0]
        self.probtopk=tf.nn.softmax(self.temp[0],axis=-1)
        self.inds=self.temp[1]
        self.next_h = next_h
        self.next_c = next_c
        self.minimize = minimize
        self.loss = loss

        self.init_op = tf.global_variables_initializer()
        self.sess.run(self.init_op)

    def reset_rnn_state(self):
        self.h = np.zeros([self.B, self.H])
        self.c = np.zeros([self.B, self.H])

    def set_rnn_state(self, h, c):
        '''
        # Arguments:
            h: np.array, shape = (B,H)
            c: np.array, shape = (B,H)
        '''
        self.h = h
        self.c = c

    def get_rnn_state(self):
        return self.h, self.c

    def predict(self, state, stateful=True,topk=0):
        '''
        Predict next action(word) probability
        # Arguments:
            state: np.array, previous word ids, shape = (B, 1)
        # Optional Arguments:
            stateful: bool, default is True
                if True, update rnn_state(h, c) to Generator.h, Generator.c
                    and return prob.
                else, return prob, next_h, next_c without updating states.
        # Returns:
            prob: np.array, shape=(B, V)
        '''
        # state = state.reshape(-1, 1)
        feed_dict = {
            self.state_in : state,
            self.h_in : self.h,
            self.c_in : self.c,
            self.topk:topk}

        if not topk:
            prob, next_h, next_c = self.sess.run(
            [self.prob, self.next_h, self.next_c],
            feed_dict)
            if stateful:
                self.h = next_h
                self.c = next_c
                return prob
            else:
                return prob, next_h, next_c
        else:
            probtopk,inds, next_h, next_c = self.sess.run(
            [self.probtopk,self.inds, self.next_h, self.next_c],
            feed_dict)
            if stateful:
                self.h = next_h
                self.c = next_c
                return [probtopk,inds]
            else:
                return [probtopk,inds], next_h, next_c        

    def update(self, state, action, reward, h=None, c=None, stateful=True):
        '''
        Update weights by Policy Gradient.
        # Arguments:
            state: np.array, Environment state, shape = (B, 1) or (B, t)
                if shape is (B, t), state[:, -1] will be used.
            action: np.array, Agent action, shape = (B, )
                In training, action will be converted to onehot vector.
                (Onehot shape will be (B, V))
            reward: np.array, reward by Environment, shape = (B, )

        # Optional Arguments:
            h: np.array, shape = (B, H), default is None.
                if None, h will be Generator.h
            c: np.array, shape = (B, H), default is None.
                if None, c will be Generator.c
            stateful: bool, default is True
                if True, update rnn_state(h, c) to Generator.h, Generator.c
                    and return loss.
                else, return loss, next_h, next_c without updating states.

        # Returns:
            loss: np.array, shape = (B, )
            next_h: (if stateful is True)
            next_c: (if stateful is True)
        '''
        if h is None:
            h = self.h
        if c is None:
            c = self.c
        state = state[:, -1].reshape(-1, 1)
        reward = reward.reshape(-1)
        feed_dict = {
            self.state_in : state,
            self.h_in : h,
            self.c_in : c,
            self.action : to_categorical(action, self.V),
            self.reward : reward}#,self.topk:0}
        _, loss, next_h, next_c = self.sess.run(
            [self.minimize, self.loss, self.next_h, self.next_c],
            feed_dict)

        if stateful:
            self.h = next_h
            self.c = next_c
            return loss
        else:
            return loss, next_h, next_c
    def top_k(self, prob, k=5):
        '''
        topk sampling uses numpy. Abandoned because of poor time efficiency.
        '''
        if not k:
            return np.random.choice(self.V, p=prob)
        def softmax(z):
            return np.exp(z) / sum(np.exp(z))

        topk = sorted([(p, i) for i, p in enumerate(prob)], reverse=True)[:k]
        k_prob = [p for p, i in topk]
        k_prob = softmax(np.log(k_prob))
        k_idx = [i for p, i in topk]
        # return k_idx, k_prob, np.random.choice(k_idx, p=k_prob)
        return np.random.choice(k_idx, p=k_prob)
    def sampling_word(self, prob,k=0):
        '''
        # Arguments:
            prob: numpy array, dtype=float, shape = (B, V),
        # Returns:
            action: numpy array, dtype=int, shape = (B, )
        '''
        # print(225,k)
        # if k == 0:
        #     k=self.V
        action = np.zeros((self.B,), dtype=np.int32)
        if k>0:
            prob,inds=prob
        else:
            ind=self.V
        for i in range(self.B):
            p = prob[i]
            if k>0:
                ind= inds[i]
            # print(231,k)
            # x,ind = top_k(p,k)

            # print(213,x.shape,ind.shape)
            action[i] = np.random.choice(ind,p=p)#, p=x)
            # action[i] = self.top_k(p,k)
            # action[i] = np.random.choice(self.V, p=p)
        return action

    def sampling_sentence(self, T, BOS=1,k=0):
        '''
        # Arguments:
            T: int, max time steps
        # Optional Arguments:
            BOS: int, id for Begin Of Sentence
        # Returns:
            actions: numpy array, dtype=int, shape = (B, T)
        '''
        # print(249,k)
        self.reset_rnn_state()
        action = np.zeros([self.B, 1], dtype=np.int32)
        action[:, 0] = BOS
        actions = action
        
        for idx in range(T):
            
            
            if idx==0:
                prob = self.predict(action,topk=0)
                action = self.sampling_word(prob,k=0).reshape(-1, 1)
            else:
                prob = self.predict(action,topk=k)
                action = self.sampling_word(prob,k=k).reshape(-1, 1)

            actions = np.concatenate([actions, action], axis=-1)
        # Remove BOS
        actions = actions[:, 1:]
        self.reset_rnn_state()
        return actions

    def generate_samples(self, T, g_data, num, output_file,k=0,use_eos=False):
        '''
        Generate sample sentences to output file
        # Arguments:
            T: int, max time steps
            g_data: SeqGAN.utils.GeneratorPretrainingGenerator
            num: int, number of sentences
            output_file: str, path
        '''
        sentences=[]

        for _ in range(num // self.B + 1):
            actions = self.sampling_sentence(T,k=k)
            actions_list = actions.tolist()

            for sentence_id in actions_list:
                if use_eos:
                    sentence=[]
                    for action in sentence_id:
                        if action==2:
                            break
                        elif action==0:
                            continue
                        else:
                            sentence.append(g_data.id2word[action] )
                
                else:
                    sentence = [g_data.id2word[action] for action in sentence_id if action != 0 and action != 2]
                sentences.append(sentence)

        output_str = ''

        for i in range(num):
            output_str += ' '.join(sentences[i]) + '\n'
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(output_str)

    def save(self, path):
        weights = []
        for layer in self.layers:
            w = layer.get_weights()
            weights.append(w)
        with open(path, 'wb') as f:
            pickle.dump(weights, f)

    def load(self, path):
        with open(path, 'rb') as f:
            weights = pickle.load(f)
        for layer, w in zip(self.layers, weights):
            layer.set_weights(w)

def Discriminator(V, E, H=64, dropout=0.1,cls=True,cls_num=2):
    '''
    Disciriminator model.
    # Arguments:
        V: int, Vocabrary size
        E: int, Embedding size
        H: int, LSTM hidden size
        dropout: float
    # Returns:
        discriminator: keras model
            input: word ids, shape = (B, T)
            output: probability of true data or not, shape = (B, 1)
    '''
    input = Input(shape=(None,), dtype='int32', name='Input')   # (B, T)
    out = Embedding(V, E, mask_zero=True, name='Embedding')(input)  # (B, T, E)
    out = Bidirectional(GRU(H,return_sequences=True))(out)
    out = Bidirectional(GRU(H))(out)
    out = Highway(out, num_layers=1)
    out = Dropout(dropout, name='Dropout')(out)
    out = Dense(1, activation='sigmoid', name='FC')(out)

    discriminator = Model(input, out)
    return discriminator

def Classifier(V, E, H=64, dropout=0.1,cls_num=2):
    input = Input(shape=(None,), dtype='int32', name='Input')   # (B, T)
    out = Embedding(V, E, mask_zero=True, name='Embedding')(input)  # (B, T, E)
    out = LSTM(H)(out)
    out = Highway(out, num_layers=1)
    out = Dropout(dropout, name='Dropout')(out)
    out = Dense(1, activation='sigmoid', name='FC')(out)
    out=Dense(cls_num, activation='softmax', name='FC2')(out)
    Classifier = Model(input, out)
    return Classifier
    
def ClassifierConv(V, E, filter_sizes, num_filters, dropout,cls_num=2):
    '''
    Another Classifier model, currently unused because keras don't support
    masking for Conv1D and it does huge influence on training.
    # Arguments:
        V: int, Vocabrary size
        E: int, Embedding size
        filter_sizes: list of int, list of each Conv1D filter sizes
        num_filters: list of int, list of each Conv1D num of filters
        dropout: float
    # Returns:
        discriminator: keras model
            input: word ids, shape = (B, T)
            output: probability of true data or not, shape = (B, 1)
    '''
    input = Input(shape=(None,), dtype='int32', name='Input')   # (B, T)
    out = Embedding(V, E, name='Embedding')(input)  # (B, T, E)
    out = VariousConv1D(out, filter_sizes, num_filters)
    out = Highway(out, num_layers=1)
    out = Dropout(dropout, name='Dropout')(out)
    out = Dense(cls_num, activation='softmax', name='FC')(out)

    classifier = Model(input, out)
    return classifier

def VariousConv1D(x, filter_sizes, num_filters, name_prefix=''):
    '''
    Layer wrapper function for various filter sizes Conv1Ds
    # Arguments:
        x: tensor, shape = (B, T, E)
        filter_sizes: list of int, list of each Conv1D filter sizes
        num_filters: list of int, list of each Conv1D num of filters
        name_prefix: str, layer name prefix
    # Returns:
        out: tensor, shape = (B, sum(num_filters))
    '''
    conv_outputs = []
    for filter_size, n_filter in zip(filter_sizes, num_filters):
        conv_name = '{}VariousConv1D/Conv1D/filter_size_{}'.format(name_prefix, filter_size)
        pooling_name = '{}VariousConv1D/MaxPooling/filter_size_{}'.format(name_prefix, filter_size)
        conv_out = Conv1D(n_filter, filter_size, name=conv_name)(x)   # (B, time_steps, n_filter)
        conv_out = GlobalMaxPooling1D(name=pooling_name)(conv_out) # (B, n_filter)
        conv_outputs.append(conv_out)
    concatenate_name = '{}VariousConv1D/Concatenate'.format(name_prefix)
    out = Concatenate(name=concatenate_name)(conv_outputs)
    return out

def Highway(x, num_layers=1, activation='relu', name_prefix=''):
    '''
    Layer wrapper function for Highway network
    # Arguments:
        x: tensor, shape = (B, input_size)
    # Optional Arguments:
        num_layers: int, dafault is 1, the number of Highway network layers
        activation: keras activation, default is 'relu'
        name_prefix: str, default is '', layer name prefix
    # Returns:
        out: tensor, shape = (B, input_size)
    '''
    input_size = K.int_shape(x)[1]
    for i in range(num_layers):
        gate_ratio_name = '{}Highway/Gate_ratio_{}'.format(name_prefix, i)
        fc_name = '{}Highway/FC_{}'.format(name_prefix, i)
        gate_name = '{}Highway/Gate_{}'.format(name_prefix, i)

        gate_ratio = Dense(input_size, activation='sigmoid', name=gate_ratio_name)(x)
        fc = Dense(input_size, activation=activation, name=fc_name)(x)
        x = Lambda(lambda args: args[0] * args[2] + args[1] * (1 - args[2]), name=gate_name)([fc, x, gate_ratio])
    return x
