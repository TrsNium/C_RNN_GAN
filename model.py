import tensorflow as tf
from module import *
import os
from util import mk_batch_func_not_pre_train, mk_batch_func_pre_train

class model():
    def __init__(self, args):
        self.args = args

    
        self.pre_train_inputs = tf.placeholder(tf.float32, [None, args.max_time_step, args.vocab_size], "pre_train_inputs")
        self.pre_train_labels = tf.placeholder(tf.float32, [None, args.max_time_step, args.vocab_size], "pre_train_labels")
        self.real = tf.placeholder(tf.float32, [None, args.max_time_step, args.vocab_size], "real_inputs")
        self.atribute_inputs = tf.placeholder(tf.float32, [None, args.atribute_size])
        

        #pre training
        self.gen = Generator(args, self.pre_train_inputs, self.atribute_inputs)
        self.p_g_loss, self.p_state = self.gen._pre_train(self.pre_train_labels)

        #train GAN
        self.fake, self.f_state = self.gen._logits()
        dis = Discriminator(args)
        dis_fake, dis_real = dis._logits(self.fake, self.real)

        self.d_loss = tf.reduce_mean(tf.squared_difference(dis_real, tf.ones_like(dis_real))) + tf.reduce_mean(tf.squared_difference(dis_fake, tf.zeros_like(dis_fake)))
        self.g_loss = tf.reduce_mean(tf.squared_difference(dis_fake, tf.ones_like(dis_fake)))

        tf.summary.scalar("pre_train_loss", self.p_g_loss)
        tf.summary.scalar("discriminator_loss", self.d_loss)
        tf.summary.scalar("generator_loss", self.g_loss)

    def train(self):
        optimizer_g_p = tf.train.AdamOptimizer(self.args.lr).minimize(self.p_g_loss)
        optimizer_g = tf.train.AdamOptimizer(self.args.lr).minimize(self.g_loss)
        optimizer_d = tf.train.AdamOptimizer(self.args.lr).minimize(self.d_loss)
        
        mk_pretrain_batch = mk_batch_func_pre_train(self.args.batch_size, self.args.max_time_step, self.args.fs)
        mk_batch = mk_batch_func_not_pre_train(self.args.batch_size, self.args.max_time_step, self.args.fs)
    
        config = tf.ConfigProto(device_count = {'GPU': 1})
        config.gpu_options.allow_growth = True
        config.log_device_placement = True
        with tf.Session(config=config) as sess:
            train_graph = tf.summary.FileWriter("./logs", sess.graph)
            merged_summary = tf.summary.merge_all()
            sess.run(tf.global_variables_initializer())
        
            if self.args.pretraining and not self.args.pretraining_done:
                print("started pre-training")
                saver_ = tf.train.Saver(tf.get_collection(tf.GraphKeys.VARIABLES, scope="Generator"))
                
                feches = {
                    "loss": self.p_g_loss,
                    "optimizer": optimizer_g_p,
                    "final_state_": self.p_state
                }

                for itr in range(self.args.pretrain_itrs):
                    inputs_, labels_ = mk_pretrain_batch(self.args.max_time_step_num)
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
                    if itr % 1000 == 0:saver_.save(sess, self.args.pre_train_path)
                print("finished pre-training")
            elif self.args.pretraining and self.pre_train_done:
                if not os.path.exists(self.args.pre_train_path):
                    print("not exits pretrain check point! damn shit byebye;)")
                    return

                saver_ = tf.train.Saver(tf.get_collection(tf.GraphKeys.VARIABLES, scope='Generator'))
                saver_.restore(sess, self.args.pre_train_path)
                print("finished restoring check point.")                
            
            saver = tf.train.Saver(tf.global_variables())
            for itr_ in range(self.args.train_itrs):
                g_loss, d_loss = [0., 0.]
                labels, atribute = mk_batch(self.args.max_time_step_num)
                state_ = sess.run(self.gen.state_)
                for step in range(self.args.max_time_step_num):
                    feed_dict = {}
                    for i, (c, h) in enumerate(self.gen.state_):
                        feed_dict[c] = state_[i].c
                        feed_dict[h] = state_[i].h
                    
                    feed_dict[self.real] = labels[:,step*self.args.max_time_step:(step+1)*self.args.max_time_step,:]
                    feed_dict[self.atribute_inputs] = atribute
                    g_loss_, state_, _ = sess.run([self.g_loss, self.gen.final_state, optimizer_g], feed_dict)
                    d_loss_, _ = sess.run([self.d_loss, optimizer_d], feed_dict)
                    g_loss += g_loss_
                    d_loss += d_loss_
                    
                g_loss /= self.args.max_time_step_num
                d_loss /= self.args.max_time_step_num
                if itr_ % 100 == 0:
                    print(itr_, ":   g_loss:", g_loss, "   d_loss:", d_loss)
                
                if itr_ % 1000 == 0:
                    saver.save(sess, self.args.train_path+"model.ckpt")
                    print("-------------------saved model---------------------")

    def generate(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.log_device_placement = True
        with tf.Session(config=config) as sess:
            saver = tf.train.Saver(tf.global_variables())
            saver.restore(sess, self.args.train_path)

            ####################生成するコードを書いたり書かなかったりウェ書いたり書かなかったりウェイよ  

