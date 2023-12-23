"""

"""
import os
import numpy as np
import pandas as pd
import sklearn
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers
import scipy.sparse as sparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

gpus = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(gpus[0], True)

# functions
def accuracy_function(real, pred):
    accuracies = tf.equal(real, tf.cast(pred>0,dtype=tf.int32))

    return accuracies

def loss_function(real, pred):

    loss_ = loss_object(real, pred)

    return tf.reduce_sum(loss_)

loss_object = tf.keras.losses.BinaryCrossentropy(
    from_logits=True,reduction='none')

def loss_function_logits(real, pred):

    loss_ = loss_object_logit(real, pred)

    return tf.reduce_sum(loss_)

loss_object_logit = tf.keras.losses.BinaryCrossentropy(
    from_logits=True,reduction='none')
train_loss = tf.keras.metrics.Mean(name='train_loss')
patient_accuracy = tf.keras.metrics.Mean(name='train_patient_accuracy')
event_accuracy = tf.keras.metrics.Mean(name='train_event_accuracy')

# the stepwise training function
def train_step(data, tar):
    # here inp is the masked-input, tar is the non-masked target
    """@data:[Tuple] of tensors, predicted from transformer, size (None,250,3...),
    @tar:[Tuple] of binary integer [next_event, patient]"""
    patient_acc = []
    event_acc = []
    loss = []
    for i in range(data.shape[0]):
        with tf.GradientTape() as tape:
            # create by batch
            predictions = pat_embd_model([data[i,0,:,:].reshape(1,250,50),
                                          data[i,1,:,:].reshape(1,250,50)],
                                         training = True)

            # this equals stochastic gradient
            event_loss = loss_function_logits(tar[i,0].reshape(-1,1),predictions[0])
            patient_loss = loss_function_logits(tar[i,1].reshape(1,), predictions[1])
            loss.append(event_loss + patient_loss)

            gradients = tape.gradient(loss, pat_embd_model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, pat_embd_model.trainable_variables))

            patient_acc.append(accuracy_function(tar[i,1].reshape(1,), predictions[1]))
            event_acc.append(accuracy_function(tar[i,0].reshape(-1,1),predictions[0]))
            if i == data.shape[0] - 1: # last step
                train_loss(loss) # log the loss and accuracy
                patient_accuracy(patient_acc)
                event_accuracy(event_acc)

# functions to create random batches for training
def create_balanced_batch_training():
    """its too big, try randomly select 2 patients and feed to it"""
    L = len(patient_event_embd)

    patient_a = []
    patient_b = []
    while (len(patient_a) < 10 or 20 < len(patient_a)) or ( len(patient_b)<10 or 20 < len(patient_b)):
    #while (20 < len(patient_a) or len(patient_a) < 2) or (20 < len(patient_b) or len(patient_b) < 2):
        random_idx = np.random.choice(L,size=2,replace=False)
        patient_a = patient_event_embd[random_idx[0]] # list of events
        patient_b = patient_event_embd[random_idx[1]] # list of events

    # try this see if it can be a batch
    patient_a = transformer.encoder(np.array(patient_a),training=False,mask=None)
    patient_b = transformer.encoder(np.array(patient_b),training=False,mask=None)
#     patient_a = tf.reshape(patient_a,shape=(len(patient_a),12500))
#     patient_b = tf.reshape(patient_b,shape=(len(patient_b),12500))
    # create training sample pairs
    data_pair = []
    target = []

    def true_case(patient):
        data_pair = []
        target = []
        for i in range(len(patient)-1):
            a,b = (patient[i].numpy(),patient[i+1].numpy())
            data_pair.append((a,b))
            target.append((1,0)) # next event, same patient
        return data_pair, target

    def random_patient_case(patient_a,patient_b):
        data_pair = []
        target = []
        for i in range(len(patient_a)):
            j = np.random.randint(len(patient_b))
            a,b = (patient_a[i].numpy(),patient_b[j].numpy())
            data_pair.append((a,b))
            target.append((0,1)) # random event, different patients
        return data_pair, target

    def shuffle_seq_case(patient):
        data_pair = []
        target = []
        j = 0
        for i in range(len(patient)):
            while j == i+1: # do not let j be the following event of i
                j = np.random.randint(0,len(patient))
            a, b = (patient[i].numpy(),patient[j].numpy())
            data_pair.append((a,b))
            target.append((0,0)) # random event, same patients
        return data_pair, target

    data,tar = true_case(patient_a)
    data_pair.extend(data)
    target.extend(tar)

    data,tar = true_case(patient_b)
    data_pair.extend(data)
    target.extend(tar)

    data,tar = random_patient_case(patient_a,patient_b)
    data_pair.extend(data)
    target.extend(tar)

    data,tar = shuffle_seq_case(patient_a)
    data_pair.extend(data)
    target.extend(tar)

    data,tar = shuffle_seq_case(patient_b)
    data_pair.extend(data)
    target.extend(tar)

    return data_pair,target

# the main training function, integrated with the stepwise training
def train(STEP):

    checkpoint_path = 'C:/Data/Lab/PatEmbedding/eMERGE/pat_embd/pool_embd/testrun/'
    ckpt = tf.train.Checkpoint(model=pat_embd_model,
                               optimizer=optimizer)

    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=1)


    # If a checkpoint exists, restore the latest checkpoint.
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('Latest checkpoint restored!!')

    # in each step, 2 random patient is selected to construct events and make predictions on those
    for step in range(STEP):
        train_loss.reset_states()
        patient_accuracy.reset_states()
        event_accuracy.reset_states()

        data_pair,target = create_balanced_batch_training()
        train_step(np.array(data_pair),np.array(target))
        if step % 20 == 0:
            print(f'Step {step + 1} Loss {train_loss.result():.4f} \
Patient Accuracy {patient_accuracy.result():.4f}, Event Accuracy {event_accuracy.result():.4f}')

    ckpt_save_path = ckpt_manager.save()

if __name__ == "__main__":

    # load the training dataset
    import multiprocessing
    import logging
    import patembd_model
    import transformer_model as model
    import json
    import pickle

    embd_data = np.load("C:/Data/Lab/PatEmbedding/eMERGE/model/50dim/50_dim_embedding/embds_mean.npy")

    with open("C:/Data/Lab/PatEmbedding/eMERGE/model/50dim/50_dim_embedding/namelist", "rb") as fp:   # Unpickling
        namelist = pickle.load(fp)

    with open("C:/Data/Lab/PatEmbedding/eMERGE/patient_vec_by_year.json") as f:
        patient_vec = json.load(f)

    def padding_sequence(seq, MAX_LENGTH=250):
        return seq + [0] * (MAX_LENGTH-len(seq))


    def create_pat_vocabulary_training(patient_vec = patient_vec):

        MAX_LENGTH = 250
        tokenizer = {voc:i+1 for i,voc in enumerate(namelist)} # 0 index is for padding
        # paired sequence (cur_event, next_event)
        patient_event_embd = []
        for pat in patient_vec.keys():
            collector = []
            for year in sorted([int(x) for x in patient_vec[pat].keys()]): # year sorted
                events = set([ x for x in patient_vec[pat][str(year)] if x in tokenizer]) # out of voc is 0
                if len(events) <= MAX_LENGTH:
                    seq = [ tokenizer[x] for x in events]
                    collector.append(padding_sequence(seq))
                else:
                    collector = []
                    break # don't collect this patient, might have issues having too many events' per year

            patient_event_embd.append(collector)

        """every pair is a patient time sliced event"""
        return patient_event_embd
    patient_event_embd = create_pat_vocabulary_training()

    # load the transformer model to get the output from it
    def load_model(savedir = "C:/Data/Lab/PatEmbedding/eMERGE/pat_embd/checkpoints/structure_L6_H_10_DIFF_2048_d200"):
        checkpoint_path = savedir
        optimizer = tf.keras.optimizers.Adam(1e-3, beta_1=0.9, beta_2=0.98,
                                             epsilon=1e-9)
        ckpt = tf.train.Checkpoint(transformer=transformer,
                                   optimizer=optimizer)

        ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=1)
        # If a checkpoint exists, restore the latest checkpoint.
        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            print('Latest checkpoint restored!!')

    # define and load the model parameter
    transformer = model.Transformer(num_layers=6, # layer, L
    d_model=200, # embeddings of  feature, H
    num_heads=10, # numbers of head A
    dff=2048,
    input_vocab_size=len(namelist),
    embedding_matrix = np.concatenate((np.zeros(shape = (1,50)),embd_data)))
    load_model()

    # start training
    # defining the patient embedding model
    pat_embd_model = patembd_model.patient_embedding_model(50)
    learning_rate = 3e-4
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                         #clipvalue = 5.0,
                                         epsilon=1e-9)

    STEP = 200
    p = multiprocessing.Process(target=train(STEP))
    p.start()
    p.join()
