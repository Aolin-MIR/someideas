from gc import callbacks
from SeqGAN.models import GeneratorPretraining, Discriminator, Generator
from SeqGAN.models import ClassifierConv as Classifier
from SeqGAN.utils import GeneratorPretrainingGenerator, DiscriminatorGenerator,ClassifierGenerator
from SeqGAN.rl import Agent, Environment
# from tensorflow.keras.optimizers import Adam
from keras.optimizers import Adam
import os
import numpy as np
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
import keras.backend as K
K.set_session(sess)

import code

class Trainer(object):
    '''
    Manage training
    '''
    def __init__(self, B, T, g_E, g_H, d_E, d_H,c_E, d_dropout, path_pos, path_neg,path_gctrain, path_label,g_lr=1e-3, d_lr=1e-3, n_sample=16, generate_samples=10000, init_eps=0.1,cls_num=2,c_filter_sizes=None,c_num_filters=None,k=0,mode='train',clip=1):
        self.B, self.T = B, T #batchsize,maxlength
        self.g_E, self.g_H = g_E, g_H
        self.d_E, self.d_H = d_E, d_H
        self.c_E=c_E
        self.c_filter_sizes=c_filter_sizes
        self.c_num_filters=c_num_filters
        self.d_dropout = d_dropout
        self.generate_samples = generate_samples
        self.g_lr, self.d_lr = g_lr, d_lr
        self.eps = init_eps
        self.init_eps = init_eps
        self.top = os.getcwd()
        self.path_pos = path_pos
        self.path_neg = path_neg
        self.path_gctrain=path_gctrain
        self.path_label=path_label
        self.cls_num=cls_num
        self.k=k
        # print(44,self.k)

        self.g_data = GeneratorPretrainingGenerator(self.path_gctrain, B=B, T=T, min_count=1) # next方法产生x, y_true数据; 都是同一个数据，比如[BOS, 8, 10, 6, 3, EOS]预测[8, 10, 6, 3, EOS]
            # self.d_data = DiscriminatorGenerator(path_pos=self.path_pos, path_neg=self.path_neg, B=self.B, shuffle=True) # next方法产生 pos数据和neg数据
        if mode=='train':
            self.c_data = ClassifierGenerator(path_pos=self.path_pos,path_label=self.path_label,B=self.B,shuffle=True)

        self.V = self.g_data.V

        self.agent = Agent(sess, B, self.V, g_E, g_H, g_lr,k)
        if mode=='train':
            self.g_beta = Agent(sess, B, self.V, g_E, g_H, g_lr,k)

            self.discriminator = Discriminator(self.V, d_E, d_H, d_dropout)
            self.classifier=Classifier(self.V, self.c_E,self.c_filter_sizes,self.c_filter_sizes,self.d_dropout,self.cls_num)
            self.env = Environment(self.discriminator,self.classifier, self.g_data, self.g_beta, n_sample=n_sample)

            self.generator_pre = GeneratorPretraining(self.V, g_E, g_H)

    def pre_train(self, g_epochs=3, d_epochs=1, g_pre_path=None ,d_pre_path=None, g_lr=1e-3, d_lr=1e-3,c_epochs=30, c_pre_path=None, c_lr=1e-3,train_g=False,train_d=False,train_c=True):
        if train_g:
            self.pre_train_generator(g_epochs=g_epochs, g_pre_path=g_pre_path, lr=g_lr)
        if train_d:
            # self.load_pre_train_g(g_pre_path)
            # self.agent.generator.generate_samples(
            # self.T, self.g_data, self.generate_samples, self.path_neg)
            self.pre_train_discriminator(d_epochs, d_pre_path, lr=d_lr)
        if train_c:
            self.pre_train_classifier( c_epochs, c_pre_path, c_lr)
    def pre_train_generator(self, g_epochs=3, g_pre_path=None, lr=1e-3):
        if g_pre_path is None:
            self.g_pre_path = os.path.join(self.top, 'data', 'save', 'generator_pre.hdf5')
        else:
            self.g_pre_path = g_pre_path

        g_adam = Adam(lr)
        self.generator_pre.compile(g_adam, 'categorical_crossentropy')
        print('Generator pre-training')
        self.generator_pre.summary()
        callback  = tf.keras.callbacks.EarlyStopping(
        monitor="loss",
        patience=5,
        restore_best_weights=True,
            )
        self.generator_pre.fit_generator(
            self.g_data,
            steps_per_epoch=None,
            epochs=g_epochs,callbacks=[callback])
        self.generator_pre.save_weights(self.g_pre_path)
        self.reflect_pre_train()

    def pre_train_discriminator(self, d_epochs=5, d_pre_path=None, lr=1e-3):
        if d_pre_path is None:
            self.d_pre_path = os.path.join(self.top, 'data', 'save', 'discriminator_pre.hdf5')
        else:
            self.d_pre_path = d_pre_path

        print('Start Generating sentences with top',self.k)
        self.agent.generator.generate_samples(self.T, self.g_data,
            self.generate_samples, self.path_neg,k=self.k)

        self.d_data = DiscriminatorGenerator(
            path_pos=self.path_pos,
            path_neg=self.path_neg,
            B=self.B,
            shuffle=True)

        d_adam = Adam(lr)
        self.discriminator.compile(d_adam, 'binary_crossentropy')
        self.discriminator.summary()
        print('Discriminator pre-training')
        callback  = tf.keras.callbacks.EarlyStopping(
                monitor="loss",
                patience=4,
                restore_best_weights=True,
                    )
        self.discriminator.fit_generator(
            self.d_data,
            steps_per_epoch=None,
            epochs=d_epochs,callbacks=[callback])
        self.discriminator.save(self.d_pre_path)
    def pre_train_classifier(self, c_epochs=30, c_pre_path=None, lr=1e-3):
        if c_pre_path is None:
            self.c_pre_path = os.path.join(self.top, 'data', 'save', 'classifier_pre.hdf5')
        else:
            self.c_pre_path = c_pre_path



        self.c_data = ClassifierGenerator(
            path_pos=self.path_pos,
            path_label=self.path_label,
            B=self.B,
            shuffle=True)

        c_adam = Adam(lr)
        self.classifier.compile(c_adam, loss=tf.keras.losses.SparseCategoricalCrossentropy())
        self.classifier.summary()
        print('Classifier pre-training')
        callback  = tf.keras.callbacks.EarlyStopping(
                monitor="loss",
                patience=4,
                restore_best_weights=True,
                    )

        self.classifier.fit_generator(
            self.c_data,
            steps_per_epoch=None,
            epochs=c_epochs,callbacks=[callback])
        self.classifier.save(self.c_pre_path)

    def load_pre_train(self, g_pre_path, d_pre_path,c_pre_path):
        self.generator_pre.load_weights(g_pre_path)
        self.reflect_pre_train()
        self.discriminator.load_weights(d_pre_path)
        self.classifier.load_weights(c_pre_path)

    def load_pre_train_g(self, g_pre_path):
        self.generator_pre.load_weights(g_pre_path)
        self.reflect_pre_train()


    def load_pre_train_d(self, d_pre_path):
        self.discriminator.load_weights(d_pre_path)

    def reflect_pre_train(self):
        i = 0
        for layer in self.generator_pre.layers:
            if len(layer.get_weights()) != 0:
                w = layer.get_weights()
                self.agent.generator.layers[i].set_weights(w)
                self.g_beta.generator.layers[i].set_weights(w)
                i += 1

    def train(self, steps=10, g_steps=5, d_steps=1, d_epochs=1,
        g_weights_path='data/save/generator.pkl',
        d_weights_path='data/save/discriminator.hdf5',
        verbose=True,
        head=1,gc=1,gd=1):
        d_adam = Adam(self.d_lr)
        self.discriminator.compile(d_adam, 'binary_crossentropy')
        self.eps = self.init_eps
        for step in range(steps):
            print("Generator training step{}".format(step))
            for index in range(g_steps):
                for _ in range(gd):
                    rewards = np.zeros([self.B, self.T])
                    self.agent.reset()
                    self.env.reset()
                    print('train g with reward from d----')
                    
                    for t in range(self.T):
                        state = self.env.get_state()
                        if t==0:
                            
                            action = self.agent.act(state, epsilon=0.0,first=True)
                        else:
                            action = self.agent.act(state, epsilon=0.0)
                        _next_state, reward, is_episode_end, _info = self.env.step(action,'d')
                        self.agent.generator.update(state, action, reward)
                        rewards[:, t] = reward.reshape([self.B, ])
                        print('reward:',np.average(rewards))
                        if is_episode_end:
                            if verbose:
                                print('Reward: {:.3f}, Episode end'.format(np.average(rewards)))
                                # self.env.render(head=head)
                            break
                for _ in range(gc):        
                    rewards = np.zeros([self.B, self.T])
                    self.agent.reset()
                    self.env.reset()
                
                    print('train g with reward from c----')
                    for t in range(self.T):
                        state = self.env.get_state()

                        if t==0:
                            
                            action = self.agent.act(state, epsilon=0.0,first=True)
                        else:
                            action = self.agent.act(state, epsilon=0.0)

                        _next_state, reward, is_episode_end, _info = self.env.step(action,'c')
                        self.agent.generator.update(state, action, reward)
                        rewards[:, t] = reward.reshape([self.B, ])
                        if np.isnan(reward).any():
                            print(226,reward)
                        print('reward:',np.average(rewards))
                        if is_episode_end:
                            if verbose:
                                print('Reward: {:.3f}, Episode end'.format(np.average(rewards)))
                                # self.env.render(head=head)
                            break
            print("Discriminator training step{}".format(step))
            # Discriminator training,IO frequecetly,to be modified.
            for _ in range(d_steps):
                self.agent.generator.generate_samples(
                    self.T,
                    self.g_data,
                    self.generate_samples,
                    self.path_neg,k=self.k)
                self.d_data = DiscriminatorGenerator(
                    path_pos=self.path_pos,
                    path_neg=self.path_neg,
                    B=self.B,
                    shuffle=True)
                self.discriminator.fit_generator(
                    self.d_data,
                    steps_per_epoch=None,
                    epochs=d_epochs)

                # self.c_data=
                # self.classifier.fit_generator()

            # Update env.g_beta to agent
            self.agent.save(g_weights_path)
            self.g_beta.load(g_weights_path)

            self.discriminator.save(d_weights_path)
            self.eps = max(self.eps*(1- float(step) / steps * 4), 1e-4)

    def save(self, g_path, d_path):
        self.agent.save(g_path)
        self.discriminator.save(d_path)

    def load(self, g_path, d_path):
        self.agent.load(g_path)
        self.g_beta.load(g_path)
        self.discriminator.load_weights(d_path)

    def test(self):
        x, y = self.d_data.next()
        pred = self.discriminator.predict(x)

        for i in range(self.B):
            txt = [self.g_data.id2word[id] for id in x[i].tolist()]

            label = y[i]
            print('{}, {:.3f}: {}'.format(label, pred[i,0], ''.join(txt)))
    
    def generate_txt(self, file_name, generate_samples,k=0,use_eos=False):
        path_neg = os.path.join(self.top, 'data', 'save', file_name)

        self.agent.generator.generate_samples(
            self.T, self.g_data, generate_samples, path_neg,k,use_eos=use_eos)

