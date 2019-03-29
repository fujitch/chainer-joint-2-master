#!/usr/bin/python
# -*- coding: utf-8 -*-
 
import lstm
import numpy as np
from chainer import optimizers, cuda
import time
import sys
import cPickle
import scipy.io
import random
import chainer
 
IN_UNITS = 40
HIDDEN_UNITS = 128
OUT_UNITS = 40
TRAINING_EPOCHS = 10000
DISPLAY_EPOCH = 10
MINI_BATCH_SIZE = 200
LENGTH_OF_SEQUENCE = 50
STEPS_PER_CYCLE = 50
NUMBER_OF_CYCLES = 100
 
xp = np
 
def compute_loss(model, sequences):
    loss = 0
    timeSeries, joints, batchSize = sequences.shape
    length_of_sequence = timeSeries
    for i in range(length_of_sequence - 1):
        x = chainer.Variable(
            xp.asarray(
                [sequences[i + 0, :, j] for j in range(batchSize)], 
                dtype=np.float32
            )
        )
        t = chainer.Variable(
            xp.asarray(
                [sequences[i + 1, :, j] for j in range(batchSize)], 
                dtype=np.float32
            )
        )
        loss += model(x, t)
    return loss 
 
# make training data
inputJoint = scipy.io.loadmat('jointDataset2.mat')
inputJoint = inputJoint['jointDataset']
inputJoint = inputJoint.astype(np.float32)
inputJoint = inputJoint.T
 
# setup model
model = lstm.LSTM(IN_UNITS, HIDDEN_UNITS, OUT_UNITS)
for param in model.params():
    data = param.data
    data[:] = np.random.uniform(-0.1, 0.1, data.shape)
 
# setup optimizer
optimizer = optimizers.Adam()
optimizer.setup(model)
 
start = time.time()
cur_start = start
for epoch in range(TRAINING_EPOCHS):
    sequences = np.ndarray((LENGTH_OF_SEQUENCE, 40, MINI_BATCH_SIZE), dtype=np.float32)
    for i in range(MINI_BATCH_SIZE):
        index = random.randint(1000*(i/10),1000*((i/10) + 1) - 100)
        sequences[:,:,i] = inputJoint[index:index+LENGTH_OF_SEQUENCE]
    model.reset_state()
    model.zerograds()
    loss = compute_loss(model, sequences)
    loss.backward()
    optimizer.update()
 
    if epoch != 0 and epoch % DISPLAY_EPOCH == 0:
        cur_end = time.time()
        # display loss
        print(
            "[{j}]training loss:\t{i}\t{k}[sec/epoch]".format(
                j=epoch, 
                i=loss.data/(sequences.shape[1] - 1), 
                k=(cur_end - cur_start)/DISPLAY_EPOCH
            )
        )
        cur_start = time.time() 
        sys.stdout.flush()
 
end = time.time()
 
# save model
cPickle.dump(model, open("./model.pkl", "wb"))
 
print("{}[sec]".format(end - start))