import tensorflow as tf
from module import *
import os
from util import *
import numpy as np
import warnings
import random


class model():
    def __init__(self, args):
        self.args = args

    
        self.pre_train_inputs = tf.placeholder(tf.float32, [None, args.max_time_step, args.vocab_size], "pre_train_inputs")
        self.pre_train_labels = tf.placeholder(tf.float32, [None, args.max_time_step, args.vocab_size], "pre_train_labels")
        self.real = tf.placeholder(tf.float32, [None, args.max_time_step, args.vocab_size], "real_inputs")
        self.atribute_inputs = tf.placeholder(tf.float32, [None, args.max_time_step, args.atribute_size+args.random_dim])
        

        #pre training
        self.gen = Generator(args, self.pre_train_inputs, self.atribute_inputs)
        self.p_g_loss, self.p_state = self.gen._pre_train(self.pre_train_labels)

        #train GAN
        self.fake, self.f_state = self.gen._logits()
        dis = Discriminator(args)
        dis_fake, dis_real = dis._logits(self.fake, self.real) 

        self.d_loss = tf.reduce_mean(-tf.log(tf.clip_by_value(dis_real, 1e-5, 1.0)) - tf.log(1 - tf.clip_by_value(dis_fake, 0.0, 1.0 - 1e-5)))
        self.g_loss = tf.reduce_mean(tf.squared_difference(dis_fake, tf.ones_like(dis_fake)))
        self.g_loss_ = tf.reduce_mean(-tf.log(tf.clip_by_value(dis_real, 1e-5, 1.0)))

        tf.summary.scalar("pre_train_loss", self.p_g_loss)
        tf.summary.scalar("discriminator_loss", self.d_loss)
        tf.summary.scalar("generator_loss", self.g_loss_)

    def train(self):
        optimizer_g_p = tf.train.GradientDescentOptimizer(self.args.lr).minimize(self.p_g_loss)
        optimizer_g = tf.train.GradientDescentOptimizer(self.args.lr).minimize(self.g_loss_)
        optimizer_d = tf.train.GradientDescentOptimizer(self.args.lr).minimize(self.d_loss)
         
        mk_pretrain_batch = mk_batch_func_pre_train(self.args.batch_size, self.args.max_time_step, self.args.fs)
        mk_batch = mk_batch_func_not_pre_train(self.args.batch_size, self.args.max_time_step, self.args.fs)
    
        config = tf.ConfigProto(device_count = {'GPU': 1})
        config.gpu_options.allow_growth = True
        config.log_device_placement = True
        with tf.Session(config=config) as sess:
            train_graph = tf.summary.FileWriter("./logs", sess.graph)
            merged_summary = tf.summary.merge_all()
            sess.run(tf.global_variables_initializer())
        
            if self.args.pretraining and not self.args.pre_train_done:
                print("started pre-training")
                saver_ = tf.train.Saver(tf.get_collection(tf.GraphKeys.VARIABLES, scope="Generator"))
                
                feches = {
                    "loss": self.p_g_loss,
                    "optimizer": optimizer_g_p,
                    "final_state_": self.p_state
                }

                for itr in range(self.args.pretrain_itrs):
                    inputs_, labels_ = mk_pretrain_batch(self.args.max_time_step_num, self.args.input_norm)
                    loss_ = 0.
                    state_ = sess.run(self.gen.state_)
                    for step in range(self.args.max_time_step_num):
                        feed_dict ={}
                        for i, (c, h) in enumerate(self.gen.state_):
                            feed_dict[c] = state_[i].c
                            feed_dict[h] = state_[i].h

                        feed_dict[self.pre_train_inputs] = inputs_[:,step*self.args.max_time_step:(step+1)*self.args.max_time_step,:]
                        feed_dict[self.pre_train_labels] = labels_[:,step*self.args.max_time_step:(step+1)*self.args.max_time_step,:]
                        vals = sess.run(feches, feed_dict)    
                        state_ = vals["final_state_"]
                        loss_ += vals["loss"]

                    if itr % 100 == 0:print("itr", itr, "     loss:",loss_/self.args.pretrain_itrs)
                    if itr % 200 == 0:saver_.save(sess, self.args.pre_train_path)
                print("finished pre-training")
            elif self.args.pretraining and self.args.pre_train_done:
                if not os.path.exists(self.args.pre_train_path):
                    print("not exits pretrain check point! damn shit byebye;)")
                    return

                saver_ = tf.train.Saver(tf.get_collection(tf.GraphKeys.VARIABLES, scope='Generator'))
                saver_.restore(sess, self.args.pre_train_path)
                print("finished restoring check point.")                
            
            saver = tf.train.Saver(tf.global_variables())
            for itr_ in range(self.args.train_itrs):
                g_loss, d_loss = [0., 0.]
                labels, atribute = mk_batch(self.args.max_time_step_num, self.args.input_norm)
                state_ = sess.run(self.gen.state_)
                for step in range(self.args.max_time_step_num):
                    feed_dict = {}
                    for i, (c, h) in enumerate(self.gen.state_):
                        feed_dict[c] = state_[i].c
                        feed_dict[h] = state_[i].h
                    
                    atribute_ = np.array([[a+[random.random() for _ in range(self.args.random_dim)] for _ in range(self.args.max_time_step)] for a in atribute])
                    feed_dict[self.real] = labels[:,step*self.args.max_time_step:(step+1)*self.args.max_time_step,:]
                    feed_dict[self.atribute_inputs] = atribute_
                    
                    g_loss_, state_, _ = sess.run([self.g_loss_, self.gen.final_state, optimizer_g], feed_dict)
                    d_loss_, _ = sess.run([self.d_loss, optimizer_d], feed_dict)
                    g_loss += g_loss_
                    d_loss += d_loss_
                    #print(sess.run(self.fake, feed_dict))
                g_loss /= self.args.max_time_step_num
                d_loss /= self.args.max_time_step_num
                if itr_ % 5 == 0:
                    print(itr_, ":   g_loss:", g_loss, "   d_loss:", d_loss)
                
                if itr_ % 20 == 0:
                    saver.save(sess, self.args.train_path+"model.ckpt")
                    print("-------------------saved model---------------------")

    def generate(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.log_device_placement = True
        with tf.Session(config=config) as sess:
            saver = tf.train.Saver(tf.global_variables())
            saver.restore(sess, self.args.train_path+"model.ckpt")

            results = []
            state_ = sess.run(self.gen.state_)
            for step in range(self.args.max_time_step_num):
                feed_dict={}
                for i, (c, h) in enumerate(self.gen.state_):
                    feed_dict[c] = state_[i].c
                    feed_dict[h] = state_[i].h
             
                atribute = np.array([[a+[random.random() for _ in range(self.args.random_dim)] for _ in range(self.args.max_time_step)] for a in [self.args.atribute_inputs]*self.args.batch_size])
                feed_dict[self.atribute_inputs] = atribute
                fake_, state_ = sess.run([self.fake, self.gen.final_state], feed_dict)
                results.append(fake_)

            results = np.transpose(np.concatenate(results, axis=1), (0,2,1)).astype(np.int16) 
            print(results.shape)
            print(np.max(results , axis=-1))
            [piano_roll_to_pretty_midi(results[i,:,:], self.args.fs, 0).write("./generated_mid/midi_{}.mid".format(i)) for i in range(self.args.batch_size)]    
            print("Done check out ./generated_mid/*.mid" )
            return np.transpose(results, (0,2,1))
