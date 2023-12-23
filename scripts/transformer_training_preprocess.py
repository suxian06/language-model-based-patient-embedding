"""
This script contains code used for constructing the transformer model to perform
the embedding for patient visits within a year.

This preprocessing includes:
1. padding sequence (L=250) with the zero token.
2. remove sparse training data (code less than 5, meaning a sequence with lengths
less than 5)
"""

# import os
#
import numpy as np
import json


with open("C:/Data/Lab/PatEmbedding/eMERGE/patient_vec_by_year.json") as f:
    patient_vec = json.load(f)

def create_code_vocabulary():
    embd_data = np.load("C:/Data/Lab/PatEmbedding/eMERGE/model/50dim/\
50_dim_embedding/embds_mean.npy")

    import pickle
    with open("C:/Data/Lab/PatEmbedding/eMERGE/model/50dim/\
50_dim_embedding/namelist", "rb") as fp:   # Unpickling
    namelist = pickle.load(fp)
    return embd_data

def padding_sequence(seq, MAX_LENGTH=250):
    return seq + [0] * (MAX_LENGTH-len(seq))

def create_pat_vocabulary_training(patient_vec = patient_vec):
    #
    MAX_LENGTH = 250
    tokenizer = {voc:i+1 for i,voc in enumerate(namelist)} # 0 index is for padding
    # paired sequence (cur_event, next_event)
    data_pair = []
    for pat in patient_vec.keys():
        collector = []
        for year in sorted([int(x) for x in patient_vec[pat].keys()]): # year sorted
            events = set([ x for x in patient_vec[pat][str(year)] if x in tokenizer])
            if len(events) <= MAX_LENGTH:
                seq = [ tokenizer[x] for x in events]
                collector.append(padding_sequence(seq))
            else:
                collector = []
                break # don't collect this patient, might have issues
        for i in range(len(collector)-1):
            data_pair.append((collector[i],collector[i+1]))

    return data_pair

def trime_training_data(train_data = train_data, train_data_input = train_data_input):
    #let at least input has 5 codes
    remove_codes = np.where([sum(train_data_input[i]) <=5 for i in range(len(train_data_input))])[0]
    idx = list(set(list(range(len(train_data)))).difference(remove_codes))
    train_data = train_data[idx]
    train_data_input = train_data_input[idx]
    del idx
    return train_data, train_data_input
