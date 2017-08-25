import pretty_midi
import os
import numpy as np
import re
import random

def read_midi_as_piano_roll(fn, fs):
    p_m = pretty_midi.PrettyMIDI(fn)
    p_r = p_m.get_piano_roll(fs)
    return np.array(p_r)

def mk_batch_func_not_pre_train(batch_size, time_step, fs=100):
    dir = ["data/"+dir for dir in os.listdir("data") if re.match("genre-", dir)]
    atribute_data_path = [path for i,path in enumerate(list(map(os.listdir,dir)))]
    atribute_size = len(dir)
    
    merged_data = []
    for i,atribute_datas in enumerate(atribute_data_path):
        for data in atribute_datas:
            merged_data.append(dir[i]+"/"+data)
    
    def mk_batch_func(max_time_step_num):
        choiced = [random.choice(merged_data) for _ in range(batch_size)]
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
    dir = ["data/"+dir for dir in os.listdir("data") if re.match("genre-", dir)]
    atribute_data_path = [path for i,path in enumerate(list(map(os.listdir,dir)))]
    atribute_size = len(dir)
    
    merged_data = []
    for i,atribute_datas in enumerate(atribute_data_path):
        for data in atribute_datas:
            merged_data.append(dir[i]+"/"+data)