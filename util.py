import pretty_midi
import os
import numpy as np
import re
import random


def piano_roll_to_pretty_midi(piano_roll, fs=100, program=0):
    '''Convert a Piano Roll array into a PrettyMidi object
     with a single instrument.
    Parameters
    ----------
    piano_roll : np.ndarray, shape=(128,frames), dtype=int
        Piano roll of one instrument
    fs : int
        Sampling frequency of the columns, i.e. each column is spaced apart
        by ``1./fs`` seconds.
    program : int
        The program number of the instrument.
    Returns
    -------
    midi_object : pretty_midi.PrettyMIDI
        A pretty_midi.PrettyMIDI class instance describing
        the piano roll.
    '''
    notes, frames = piano_roll.shape
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=program)

    # pad 1 column of zeros so we can acknowledge inital and ending events
    piano_roll = np.pad(piano_roll, [(0, 0), (1, 1)], 'constant')

    # use changes in velocities to find note on / note off events
    velocity_changes = np.nonzero(np.diff(piano_roll).T)

    # keep track on velocities and note on times
    prev_velocities = np.zeros(notes, dtype=int)
    note_on_time = np.zeros(notes)

    for time, note in zip(*velocity_changes):
        # use time + 1 because of padding above
        velocity = piano_roll[note, time + 1]
        time = time / fs
        if velocity > 0:
            if prev_velocities[note] == 0:
                note_on_time[note] = time
                prev_velocities[note] = velocity
        else:
            pm_note = pretty_midi.Note(
                velocity=prev_velocities[note],
                pitch=note,
                start=note_on_time[note],
                end=time)
            instrument.notes.append(pm_note)
            prev_velocities[note] = 0
    pm.instruments.append(instrument)
    return pm

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
    
    def mk_batch_func(max_time_step_num, norm=True):
        r = []
        atribute = []
        for _ in range(batch_size):
            p_r = None
            while True:
                try:
                    path = random.choice(merged_data)
                    p_r = read_midi_as_piano_roll(path, fs)
                    if max_time_step_num*time_step > p_r.shape[1]:
                        print(p_r.shape)
                        continue
                    break
                except:
                    continue
            
            p_r = p_r/np.max(p_r) if norm else p_r/np.max(p_r) *127

            r.append(p_r[:,:time_step*max_time_step_num])
            init_ =  [0]*atribute_size
            init_[dir.index("/".join(path.split("/")[:2]))] = 1
            atribute.append(init_)
    
        return np.transpose(np.array(r),(0,2,1)), atribute
    
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

    def mk_batch_func(max_time_step_num, norm=True):
        x = []
        label = []
        for _ in range(batch_size):
            p_r = None
            while True:
                try:
                    path = random.choice(merged_data)
                    p_r = read_midi_as_piano_roll(path, fs)
                    if max_time_step_num*time_step+1> p_r.shape[1]:
                        continue
                    break
                except:
                    continue

            p_r = p_r/np.max(p_r) if norm else p_r/np.max(p_r) *127

            x.append(p_r[:,:time_step*max_time_step_num])
            label.append(p_r[:,:time_step*max_time_step_num])
        return np.transpose(np.array(x), (0,2,1)).astype(np.float32), np.transpose(np.array(label), (0,2,1)).astype(np.float32)

    return mk_batch_func

def n_sigmoid(x):
    return 1/1-np.exp(-x)
