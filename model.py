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
        dis = Discriminator(args)
        self.gen = Generator(args, self.pre_train_inputs, self.atribute_inputs)
        self.p_g_loss, self.p_state, self.p_out = self.gen._pre_train(self.pre_train_labels)
        p_d_logits = dis._logits(self.pre_train_labels, None, True, False)
        self.p_d_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=p_d_logits, labels=tf.ones_like(p_d_logits)))
        
        #train GAN
        self.fake, self.f_state = self.gen._logits()
        dis_fake, dis_real = dis._logits(self.fake, self.real, reuse=True) 
        dis_fake = tf.reshape(dis_fake, [-1, 1])
        dis_real = tf.reshape(dis_real, [-1, 1])

        self.d_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=dis_real, labels=tf.ones_like(dis_real)) + tf.nn.sigmoid_cross_entropy_with_logits(logits=dis_fake, labels=tf.zeros_like(dis_fake)))
        self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=dis_fake, labels=tf.ones_like(dis_fake)))

        #tf.summary.scalar("pre_train_loss", self.p_g_loss)
        tf.summary.scalar("discriminator_loss", self.d_loss)
        tf.summary.scalar("generator_loss", self.g_loss)

    def train(self):
        optimizer_g_p = tf.train.GradientDescentOptimizer(self.args.lr).minimize(self.p_g_loss)
        optimizer_d_p = tf.train.GradientDescentOptimizer(self.args.lr).minimize(self.p_d_loss)
        optimizer_g = tf.train.GradientDescentOptimizer(self.args.lr).minimize(self.g_loss)
        optimizer_d = tf.train.GradientDescentOptimizer(self.args.d_lr).minimize(self.d_loss)
         
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
                saver_ = tf.train.Saver(tf.global_variables())
                
                feches = {
                    "g_loss": self.p_g_loss,
                    "d_loss": self.p_d_loss,
                    "optimizer_g": optimizer_g_p,
                    "optimizer_d": optimizer_d_p,
                    "final_state_": self.p_state,
                    "out": self.p_out
                }

                for itr in range(self.args.pretrain_itrs):
                    inputs_, labels_ = mk_pretrain_batch(self.args.max_time_step_num, self.args.input_norm)
                    g_loss_ = 0.
                    d_loss_ = 0.
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
                        g_loss_ += vals["g_loss"]
                        d_loss_ += vals["d_loss"]
                        out = vals["out"]
                        out[out > 127] = 127
                        out = np.transpose(out, (0,2,1)).astype(np.int16) 
                        #print(np.max(out, axis=1))
                        [piano_roll_to_pretty_midi(out[i,:,:], self.args.fs, 0).write("./generated_mid/p_midi_{}.mid".format(i)) for i in range(self.args.batch_size)] 
                    if itr % 100 == 0:print("itr", itr, "     g_loss:",g_loss_/self.args.pretrain_itrs,"     d_loss:",d_loss_/self.args.pretrain_itrs)
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
                    
                    g_loss_, state_, _ = sess.run([self.g_loss, self.gen.final_state, optimizer_g], feed_dict)
                    d_loss_, _, summary = sess.run([self.d_loss, optimizer_d, merged_summary], feed_dict)
                    g_loss += g_loss_
                    d_loss += d_loss_
                    #print(sess.run(self.fake, feed_dict))
                g_loss /= self.args.max_time_step_num
                d_loss /= self.args.max_time_step_num
                if itr_ % 5 == 0:
                    print(itr_, ":   g_loss:", g_loss, "   d_loss:", d_loss)
                    train_graph.add_summary(summary, itr_)

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
                print(atribute)
                fake_, state_ = sess.run([self.fake, self.gen.final_state], feed_dict)
                results.append(fake_)

            results = np.transpose(np.concatenate(results, axis=1), (0,2,1)).astype(np.int16) 
            print(results.shape)
            print(np.max(results , axis=-1))
            [piano_roll_to_pretty_midi(results[i,:,:]*127, self.args.fs, 0).write("./generated_mid/midi_{}.mid".format(i)) for i in range(self.args.batch_size)]    
            print("Done check out ./generated_mid/*.mid" )
            return np.transpose(results, (0,2,1))
