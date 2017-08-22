import tensorflow as tf

class module(object):
    def define_cell(args):
        cell_ = tf.contrib.rnn.BasicLSTMCell(args.dis_rnn_size, tf.get_variable_scope().reuse)
        if args.gen_keep_prob < 1.:
            cell_ = tf.contrib.rnn.DropoutWrapper(cell_, output_keep_prob=args.gen_keep_prob)
        return cell_

class Generator(module):
    def __init__(self, args, name="Genenator"):
        self.y = y
        with tf.variable_scope(name) as scope:
            scope.set_regularizer(tf.contrib.layers.l2_regularizer(scale=FLAGS.reg_scale))
            cell_ = tf.contrib.rnn.MultiRNNCell([super(self).define_cell() for _ in range(args.num_layers_d)], state_is_tuple = True)
            random_ = tf.random_uniform(shape=[args.batch_size, args.max_time_step, args.feature_size], minval=0.0, maxval=1.0, dtype=data_type())
                
            state_ = cell_.zero_state(batch_size=args.batch_size, dtype=tf.float32)
            self.outputs = []
            for t_ in range(args.max_time_step):
                if t_ != 0:
                    scope.reuse_variables()

                rnn_input_ = tf.layers.dense(random_[:,t_,:], args.rnn_input_size, tf.nn.relu, name="RNN_INPUT_DENSE")
                rnn_output_, state_ = cell_(rnn_input_, state_)
                output_ = tf.layers.dense(rnn_output_, args.vocab_size, name="RNN_OUT_DENSE")
                self.outputs.append(output_)

            scope.reuse_variables()

            state_ = cell_.zero_state(batch_size=args.batch_size, dtype=tf.float32)
            self.pre_train_outputs = []
            for t_ in range(args.max_time_step):
                if t_ != 0:
                    scope.reuse_variables()

                rnn_input_ = tf.layers.dense(random_[:,t_,:], args.rnn_input_size, tf.nn.relu, name="RNN_INPUT_DENSE")
                rnn_output_, state_ = call_(rnn_input_, state)
                output_ = tf.layers.dense(rnn_output_, args.vocab_size, name="RNN_OUT_DENSE")
                self.pre_train_outputs.append(output_)
            
            reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            self.reg_loss = args.reg_constant * sum(reg_losses)
    
    def _pre_train(self, y):
        loss = tf.reduce_mean(tf.squared_difference(self.pre_train_outputs, y)) + self.reg_loss
        return loss

    def _logits(self):
        return self.outputs

def Discriminator(module):
    def __init__(self, args, name="Discriminator"):
        self.name = name
        self.args = args

    def _logits(self, x, reuse=False) as scope:
        with tf.variable_scope(self.name) as scope():
            if reuse:
                scope.reuse_variables()

            fw = tf.contrib.rnn.MultiRNNCell([super(self).define_cell() for _ in range(self.args.num_layers_d)], state_is_tuple=True)
            bw = tf.contrib.rnn.MultiRNNCell([super(self).define_cell() for _ in range(self.args.num_layers_d)], state_is_tuple=True)
            rnn_output, _, _ = tf.nn.bidirectional_dynamic_rnn(fw,
                                            bw,
                                            x,
                                            initial_state_fw=fw.zero_state(batch_size=self.args.batch_size, dtype=tf.float32),
                                            initial_state_bw=bw.zero_state(batch_size=self.args.batch_size, dtype=tf.float32), 
                                            dtype=tf.float32,
                                            swap_memory=True)
            
            outputs = [] 
            for t_ in nange(self.args.max_time_step):
                if != 0:
                    scope.reuse_variables()

                outputs.append(tf.layers.dense(tf.concat([rnn_output[0][:,t_,:], rnn_output[1][:,t_,:]], axis=-1), 1, name="RNN_OUTPUT_DENSE"))
            logits = tf.transpose(tf.stack(outputs), (1,0,2))
        return logits
    
