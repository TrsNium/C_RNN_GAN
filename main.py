import argparse
import os
from model import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--lr", dest="lr", type=float, default= 0.002)
    parser.add_argument("--train", dest="train", type=bool, default=True)
    parser.add_argument("--train_itr", dest="train_itrs", type=int, default=1000)
    parser.add_argument("--pretraining", dest="pretraining", type=bool, default=False)
    parser.add_argument("--pretraining_done", dest="pretraining_done", type=bool, default=False)
    parser.add_argument("--fs", dest="fs", type=int, default=100)
    parser.add_argument("--atribute_size", dest="atribute_size", type=int, default=3)
    parser.add_argument("--pretrain_itrs", dest="pretrain_itrs", type=int, default=10000)
    parser.add_argument("--batch_size", dest="batch_size", type=int, default=10)
    parser.add_argument("--max_time_step", dest="max_time_step", type=int, default=200)
    parser.add_argument("--vocab_size", dest="vocab_size", type=int, default=2000)
    parser.add_argument("--pre_train_path", dest="pre_train_path", type=str, default="./saved_pre_train/")
    parser.add_argument("--max_time_step_num", dest="max_time_step_num", type=int, default=10)
    parser.add_argument("--train_path", dest="train_path", type=str, default="./train_path/")
    args = parser.parse_args()

    if not os.path.exists(args.train_path):
        os.mkdir(args.train_path)

    if not os.path.exists(args.pre_train_path):
        os.mkdir(args.pre_train_path)

    if not os.path.exists("./logs/"):
        os.mkdir("./logs/")

    model_ = model(args) 
    if args.train():
        model_.train()
