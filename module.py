import tensorflow as tf


def define_cell(rnn_size, keep_prob):
    cell_ = tf.contrib.rnn.BasicLSTMCell(rnn_size, state_is_tuple=True, reuse=tf.get_variable_scope().reuse)
    if keep_prob < 1.:
        cell_ = tf.contrib.rnn.DropoutWrapper(cell_, output_keep_prob=keep_prob)
    return cell_

class Generator():
    def __init__(self, args, x=None, attribute=None, name="Genenator"):
        with tf.variable_scope(name) as scope:
            scope.set_regularizer(tf.contrib.layers.l2_regularizer(scale=args.scale))
            cell_ = tf.contrib.rnn.MultiRNNCell([define_cell(args.gen_rnn_size, args.keep_prob) for _ in range(args.num_layers_g)])
         
            self.state_ = cell_.zero_state(batch_size=args.batch_size, dtype=tf.float32)
            outputs = []
            for t_ in range(args.max_time_step):
                if t_ != 0:
                    scope.reuse_variables()

                rnn_input_ = tf.layers.dense(attribute, args.gen_rnn_input_size, tf.nn.relu, name="RNN_INPUT_DENSE")
                _ = tf.layers.dense(x, args.gen_rnn_input_size, tf.nn.relu, name="RNN_PRE_INPUT_DENSE")
                rnn_output_, state_ = cell_(rnn_input_, self.state_)
                output_ = tf.layers.dense(rnn_output_, args.vocab_size, name="RNN_OUT_DENSE")
                outputs.append(output_)
       
            self.final_state = self.state_
            self.outputs = tf.transpose(tf.stack(outputs), (1,0,2))
            scope.reuse_variables()

            ##pre training
            self.state_ = cell_.zero_state(batch_size=args.batch_size, dtype=tf.float32)
            pre_train_outputs = []
            for t_ in range(args.max_time_step):
                if t_ != 0:
                    scope.reuse_variables()

                rnn_input_ = tf.layers.dense(x[:,t_,:], args.gen_rnn_input_size, tf.nn.relu, name="RNN_PRE_INPUT_DENSE")
                rnn_output_, state_ = cell_(rnn_input_, self.state_)
                output_ = tf.layers.dense(rnn_output_, args.vocab_size, name="RNN_OUT_DENSE")
                pre_train_outputs.append(output_)

            self.p_state = self.state_
            self.pre_train_outputs = tf.transpose(tf.stack(pre_train_outputs), (1,0,2)) 
            reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            self.reg_loss = args.reg_constant * sum(reg_losses)
    
    def _pre_train(self, y):
        loss = tf.reduce_mean(tf.squared_difference(self.pre_train_outputs, y)) + self.reg_loss
        return loss, self.p_state

    def _logits(self):
        return self.outputs, self.state_

class Discriminator(object):
    def __init__(self, args, name="Discriminator"):
        self.name = name
        self.args = args

    def _logits(self, x, y):
        with tf.variable_scope(self.name) as scope:

            fw = tf.contrib.rnn.MultiRNNCell([define_cell(self.args.dis_rnn_size, self.args.keep_prob) for _ in range(self.args.num_layers_d)], state_is_tuple=True)
            bw = tf.contrib.rnn.MultiRNNCell([define_cell(self.args.dis_rnn_size, self.args.keep_prob) for _ in range(self.args.num_layers_d)], state_is_tuple=True)
            rnn_output, state = tf.nn.bidirectional_dynamic_rnn(fw,
                                                                bw,
                                                                x,
                                                                initial_state_fw=fw.zero_state(batch_size=self.args.batch_size, dtype=tf.float32),
                                                                initial_state_bw=bw.zero_state(batch_size=self.args.batch_size, dtype=tf.float32), 
                                                                dtype=tf.float32,
                                                                swap_memory = True)
            
            outputs = [] 
            for t_ in range(self.args.max_time_step):
                if t_ != 0:
                    scope.reuse_variables()

                outputs.append(tf.layers.dense(tf.concat([rnn_output[0][:,t_,:], rnn_output[1][:,t_,:]], axis=-1), 1, name="RNN_OUTPUT_DENSE"))
            x_logits = tf.transpose(tf.stack(outputs), (1,0,2))
        
            scope.reuse_variables()
            rnn_output, state = tf.nn.bidirectional_dynamic_rnn(fw,
                                                                bw,
                                                                y,
                                                                initial_state_fw=fw.zero_state(batch_size=self.args.batch_size, dtype=tf.float32),
                                                                initial_state_bw=bw.zero_state(batch_size=self.args.batch_size, dtype=tf.float32),
                                                                dtype=tf.float32,
                                                                swap_memory=True)
            outputs = []
            for t_ in range(self.args.max_time_step):
                if t_ != 0:
                    scope.reuse_variables()

                outputs.append(tf.layers.dense(tf.concat([rnn_output[0][:,t_,:],rnn_output[1][:,t_,:]], axis=-1), 1, name="RNN_OUTPUT_DENSE"))
            y_logits = tf.transpose(tf.stack(outputs), (1,0,2))
            return x_logits, y_logits

    
