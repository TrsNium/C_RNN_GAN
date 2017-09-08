import argparse
import os
from model import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--lr", dest="lr", type=float, default= 0.08)
    parser.add_argument("--d_lr", dest="d_lr", type=float, default=0.01)
    parser.add_argument("--train", dest="train", type=bool, default=False)
    parser.add_argument("--train_itr", dest="train_itrs", type=int, default=1000001)
    parser.add_argument("--pretraining", dest="pretraining", type=bool, default=True)
    parser.add_argument("--pre_train_done", dest="pre_train_done", type=bool, default=False)
    parser.add_argument("--fs", dest="fs", type=int, default=1)
    parser.add_argument("--atribute_size", dest="atribute_size", type=int, default=3)
    parser.add_argument("--pretrain_itrs", dest="pretrain_itrs", type=int, default=3001)
    parser.add_argument("--batch_size", dest="batch_size", type=int, default=1)
    parser.add_argument("--max_time_step", dest="max_time_step", type=int, default=120)
    parser.add_argument("--vocab_size", dest="vocab_size", type=int, default=128)
    parser.add_argument("--pre_train_path", dest="pre_train_path", type=str, default="./saved_pre_train/model.ckpt")
    parser.add_argument("--max_time_step_num", dest="max_time_step_num", type=int, default=1)
    parser.add_argument("--train_path", dest="train_path", type=str, default="./train_path/")
    parser.add_argument("--scale", dest="scale", type=float, default=1.)
    parser.add_argument("--num_layers_g", dest="num_layers_g", type=int, default=3)
    parser.add_argument("--num_layers_d", dest="num_layers_d", type=int, default=3)
    parser.add_argument("--gen_rnn_size", dest="gen_rnn_size", type=int, default=312)
    parser.add_argument("--dis_rnn_size", dest="dis_rnn_size", type=int, default=312)
    parser.add_argument("--keep_prob", dest="keep_prob", type=float, default=0.56)
    parser.add_argument("--gen_rnn_input_size", dest="gen_rnn_input_size", type=int, default=128)
    parser.add_argument("--reg_constant", dest="reg_constant", type=float, default=0.5)
    parser.add_argument("--atribute_inputs", dest="atribute_inputs", type=list, default=[0,1,0])
    parser.add_argument("--input_norm", dest="input_norm", type=bool, default=False)
    parser.add_argument("--random_dim", dest="random_dim", type=int, default=32)
    args = parser.parse_args()

    if not os.path.exists("generated_mid"):
        os.mkdir("generated_mid")

    if not os.path.exists(args.train_path):
        os.mkdir(args.train_path)

    if not os.path.exists(args.pre_train_path):
        os.mkdir(args.pre_train_path)

    if not os.path.exists("./logs/"):
        os.mkdir("./logs/")

    model_ = model(args) 
    if args.train:
        model_.train()
    else:
        c = model_.generate()
        with open("visualize.txt", "w") as fs:
            c = c.tolist()[0]
            sentence = ""
            for i in range(len(c)):
                sentence_ = ",".join(list(map(str ,c[i]))) + "/n"
                sentence+=sentence_
            fs.write(sentence)
