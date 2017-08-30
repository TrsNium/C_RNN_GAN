import pretty_midi
import os
import numpy as np
import re
import random

def read_midi_as_piano_roll(fn, fs):
    p_m = pretty_midi.PrettyMIDI(fn)
    p_r = p_m.get_piano_roll(fs)
    return np.array(p_r)

def mk_index():
    dir = ["data/"+dir for dir in os.listdir("data") if re.match("genre-", dir)]
    content = "\n".join(dir)
    with open("data/index.txt", "w") as fs:
        fs.write(content)

def read_index():
    with open("data/index.txt", "r") as fs:
        lines = fs.readlines()
    return [line.split("\n")[0] for line in lines]

def mk_batch_func_not_pre_train(batch_size, time_step, fs=100):
    if not os.path.exists("data/index.txt"):
        mk_index()
        dir = read_index()
    else:
        dir = read_index()

    atribute_data_path = [path for i,path in enumerate(list(map(os.listdir,dir)))]
    atribute_size = len(dir)
    
    merged_data = []
    for i,atribute_datas in enumerate(atribute_data_path):
        for data in atribute_datas:
            merged_data.append(dir[i]+"/"+data)
    
    def mk_batch_func(max_time_step_num):
        r = []
        atribute = []
        for _ in range(batch_size):
            p_r = None
            while p_r == None:
                try:
                    path = random.choice(merged_data)
                    p_r = read_midi_as_piano_roll(path, fs)
                except:
                    continue
            
            p_r /= np.max(p_r)
            r.append(p_r[:,:time_step*max_time_step_num])
            init_ =  [0]*atribute_size
            init_[dir.index("/".join(path.split("/")[:2]))] = 1
            atribute.append(init_)
    
        return np.transpose(np.array(r),(0,2,1)), np.array(atribute)
    
    return mk_batch_func
    
def mk_batch_func_pre_train(batch_size, time_step, fs=100):
    if not os.path.exists("data/index.txt"):
        mk_index()
        dir = read_index()
    else:
        dir = read_index()
    
    atribute_data_path = [path for i,path in enumerate(list(map(os.listdir,dir)))]
    atribute_size = len(dir)
    
    merged_data = []
    for i,atribute_datas in enumerate(atribute_data_path):
        for data in atribute_datas:
            merged_data.append(dir[i]+"/"+data)

    def mk_batch_func(max_time_step_num):
        x = []
        label = []
        for _ in range(batch_size):
            p_r = None
            while p_r == None:
                try:
                    path = random.choice(merged_data)
                    p_r = read_midi_as_piano_roll(path, fs)
                except:
                    continue

            p_r /= np.max(p_r)
            x.append(p_r[:,:time_step*max_time_step_num])
            label.append(p_r[:,1:time_step*max_time_step_num+1])
        return np.transpose(np.array(x), (0,2,1)), np.transpose(np.array(label), (0,2,1))

    return mk_batch_func

