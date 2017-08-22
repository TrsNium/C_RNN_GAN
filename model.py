import tensorflow as tf
from module import *
import os


class model():
    def __ini__(self, args):
        self.args.args
        self.pre_train_labels = tf.placeholder(tf.float32, [None, args.time_step, args.vocab_size], "pre_train_labels")
        self.real = tf.placeholder(tf.float32, [None, args.max_time_step, args.vocab_size], "real_inputs")
        

        #pre training
        gen= Generator(args)
        gen.pre_train_outputs()
        self.p_g_loss = gen._pre_train(self.pre_train_labels)

        #train GAN
        self.fake = gen._logits()
        dis = Discriminator(args)
        dis_real = dis._logits(self.real, reuse=False)
        dis_fake = dis._logits(self.fake, reuse=True)

        self.d_loss = tf.reduce_mean(tf.squared_difference(dis_real, tf.ones_like(dis_real))) + tf.reduce_mean(tf.squared_difference(dis_fake, tf.zeros_like(dis_fake)))
        self.g_loss = tf.reduce_mean(tf.squared_difference(dis_fake, tf.ones_like(dis_fake)))

        with tf.variable_scope("summary"):
            tf.summary.scalar("pre_train_loss", self.p_g_loss)
            tf.summary.scalar("discriminator_loss", self.d_loss)
            tf.summary.scalar("generator_loss", self.g_loss)

    def train(self):
        optimizer_g_p = tf.train.AdamOptimizer(self.args.lr).minimize(self.p_g_loss)
        optimizer_g = tf.train.AdamOptimizer(self.args.lr).minimize(self.g_loss)
        optimizer_d = tf.train.AdamOptimizer(self.args.lr).minimize(self.d_loss)

        config = tf.ConfigProto()
        with tf.Session(config=config) as sess:
            train_graph = tf.summary.FileWriter("./logs", sess.graph)
            merged_summary = tf.summary.merge_all()
            sess.run(tf.global_variables_initializer())
        
            if self.args.pretraining:
                print("started pre-training")
                saver_ = tf.train.Saver(tf.get_collection(tf.GraphKeys.VARIABLES, scope="Generater")))
                for itr in range(self.args.pretrain_itrs):
                    pass

                saver_.save(sess, self.args.pretrain_path)
                print("finished pre-training")
            else:
                if not os.path.exists(self.args.pretrain_path):
                    print("not exits pretrain check point! damn shit byebye;)")
                    return

                saver_ = tf.train.Saver(tf.get_collection(tf.GraphKeys.VARIABLES, scope='Generater'))
                saver_.restore(sess, self.args.pretrain_path)
                print("finished restoring check point.")

            saver = tf.train.Saver(tf.global_variables())
            for itr_ in range(self.args.train_itrs):

                
                if itr_ % 100 == 0:
                    train_graph.add_summary(summary, itr_)

                if itr_ % 1000 == 0:
                    saver.save(sess, self.args.train_path)

    def generate(self):
        config = tf.ConfigProto()
        with tf.Session(config=config) as sess:
            saver = tf.train.Saver(tf.global_variables())
            saver.restore(sess, self.args.train_path)

            ####################生成するコードを書いたり書かなかったりウェ書いたり書かなかったりウェイよ  

