"""
This script contains code used for constructing the transformer model to perform
the embedding for patient visits within a year.
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as sp
import seaborn as sns

import sklearn
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers
import scipy.sparse as sparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

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
