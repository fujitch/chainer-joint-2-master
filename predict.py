#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cPickle
import numpy as np
from chainer import optimizers, cuda
import chainer
import scipy.io
import random
 
 
MODEL_PATH = "./model10000.pkl"
PREDICTION_LENGTH = 25
PREDICTION_PATH = "./prediction.txt"
INITIAL_PATH = "./initial.txt"
MINI_BATCH_SIZE = 60
LENGTH_OF_SEQUENCE = 50
STEPS_PER_CYCLE = 50
NUMBER_OF_CYCLES = 100
xp = np
 
 
def predict_sequence(model, input_seq, output_seq, dummy):
    sequences_col = len(input_seq)
    model.reset_state()
    for i in range(sequences_col):
        print(i)
        x = chainer.Variable(xp.asarray(input_seq[i:i+1], dtype=np.float32))
        future = model(x, dummy)
    cpu_future = chainer.cuda.to_cpu(future.data)
    return cpu_future
 
 
def predict(seq, model, pre_length, initial_path, prediction_path):
    # initial sequence 
    input_seq = np.array(seq[:seq.shape[0]/2])
 
    output_seq = np.empty(0)
 
    # append an initial value
    output_seq = np.append(output_seq, input_seq[-1])
    output_seq = output_seq[np.newaxis, :]
 
    model.train = False
    dummy = chainer.Variable(xp.asarray(np.zeros((1,40)), dtype=np.float32))
 
    for i in range(pre_length):
        future = predict_sequence(model, input_seq, output_seq, dummy)
        input_seq = np.delete(input_seq, [0], 0)
        input_seq = np.r_[input_seq, future]
        output_seq = np.r_[output_seq, future]
 
    with open(prediction_path, "w") as f:
        for (i, v) in enumerate(output_seq.tolist(), start=input_seq.shape[0]):
            f.write("{i} {v}\n".format(i=i-1, v=v))
 
    with open(initial_path, "w") as f:
        for (i, v) in enumerate(seq.tolist()):
            f.write("{i} {v}\n".format(i=i, v=v))
            
# load model
model = cPickle.load(open(MODEL_PATH))
 
# make data
inputJoint = scipy.io.loadmat('forTestJointDataset.mat')
inputJoint = inputJoint['jointDataset']
inputJoint = inputJoint.astype(np.float32)
inputJoint = inputJoint.T
sequences = np.ndarray((LENGTH_OF_SEQUENCE, 40, MINI_BATCH_SIZE), dtype=np.float32)
for i in range(MINI_BATCH_SIZE):
    sequences[:,:,i] = inputJoint[3300:3350]
sample_index = 45
predict(sequences[:, :, sample_index], model, PREDICTION_LENGTH, INITIAL_PATH, PREDICTION_PATH)