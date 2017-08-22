import argparse
import os
from model import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")


    args = parser.parse_args()

    model_ = model(args) 
    if args.train():
        model_.train()
