#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Last Modified: 2024-11-13
# Modified By: H. Kang

import tensorflow as tf
from keras import backend as K
import random
import pickle
from keras.utils import np_utils
from keras.optimizers import adamax_v2
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from argparse import ArgumentParser
import numpy as np
import os
import gc
import logging
from datetime import datetime
import sys
from utility import LoadDataNoDefCW
from Model_NoDef import DFNet


PARSER = ArgumentParser()
PARSER.add_argument("--standard_scaler", action="store_true", help="Apply Standard Scaler")
PARSER.add_argument("--quantile_trans", action="store_true", help="Apply Quantile Transformer")
PARSER.add_argument("--use_early_stopping", action="store_true", help="Use Early Stopping during training.")
PARSER.add_argument("-g", type=str, default='7', help="GPU device to use.")
PARSER.add_argument("-e", type=int, default=100, help="Number of training epochs.")
args = PARSER.parse_args()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = args.g

# parameter settings
FEATURE = "ipd"
SCENARIO = "ff_sl"
NB_EPOCH = args.e
BATCH_SIZE = 128
LENGTH = 5000
NB_CLASSES = 100
INPUT_SHAPE = (LENGTH, 1)
OPTIMIZER = adamax_v2.Adamax(lr=0.002)
LOG_FULLPATH = ""

class LoggerWriter:
    def __init__(self, logger, level):
        self.logger = logger
        self.level = level

    def write(self, message):
        if message.strip():
            self.level(message)

    def flush(self):
        pass

def get_log_filename(feature, epoch, standard_scaler, quantile_trans, early_stopping):
    suffix = ""
    if standard_scaler:
        suffix += "_SS"
    if quantile_trans:
        suffix += "_QT"
    if early_stopping:
        suffix += "_ES"
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{feature}_{epoch}{suffix}_{current_time}.txt"

def create_log(log_path, feature, epoch, standard_scaler, quantile_trans, early_stopping):
    global LOG_FULLPATH

    log = logging.getLogger()
    log.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    log.addHandler(stream_handler)

    log_filename = get_log_filename(feature, epoch, standard_scaler, quantile_trans, early_stopping)
    LOG_FULLPATH = os.path.join(log_path, log_filename)

    file_handler = logging.FileHandler(LOG_FULLPATH, 'w')
    file_handler.setFormatter(formatter)
    log.addHandler(file_handler)

    print(f"Log is being saved at: {LOG_FULLPATH}")
    log.info(f"Log files are being saved at: {LOG_FULLPATH}")
    return log

log_path = f"/scratch4/kanghosung/starlink_DF/log/{FEATURE}"
if not os.path.exists(log_path):
    os.makedirs(log_path, exist_ok=True)

log = create_log(log_path, FEATURE, NB_EPOCH, args.standard_scaler, args.quantile_trans, args.use_early_stopping)
sys.stdout = LoggerWriter(log, log.info)

# Function to load the data
def load_data(feature, scenario, standard_scaler=False, quantile_trans=False):
    root = f"/scratch4/kanghosung/starlink_DF/f_ex/{feature}"
    tr_path = os.path.join(root, f"{scenario}_{feature}_training_56inst.npz")
    val_path = os.path.join(root, f"{scenario}_{feature}_valid_12inst.npz")
    te_path = os.path.join(root, f"{scenario}_{feature}_testing_12inst.npz")

    train = np.load(tr_path)
    valid = np.load(val_path)
    test = np.load(te_path)
    
    X_train, y_train = train['data'], train['labels']
    X_valid, y_valid = valid['data'], valid['labels']
    X_test, y_test = test['data'], test['labels']

    # Apply scaling or transformation if specified
    if standard_scaler:
        print("Applying Standard Scaler...")
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_valid = scaler.transform(X_valid)
        X_test = scaler.transform(X_test)
    elif quantile_trans:
        print("Applying Quantile Transformer...")
        transformer = QuantileTransformer(output_distribution='normal') # rm random 
        X_train = transformer.fit_transform(X_train)
        X_valid = transformer.transform(X_valid)
        X_test = transformer.transform(X_test)

    del train, valid, test
    gc.collect()
    return X_train, y_train, X_valid, y_valid, X_test, y_test

#Load data
X_train, y_train, X_valid, y_valid, X_test, y_test = load_data(FEATURE, SCENARIO, args.standard_scaler, args.quantile_trans)


X_train = X_train.astype('float32')[:, :, np.newaxis]
X_valid = X_valid.astype('float32')[:, :, np.newaxis]
X_test = X_test.astype('float32')[:, :, np.newaxis]
# Convert labels to one-hot encoding
y_train = np_utils.to_categorical(y_train, NB_CLASSES)
y_valid = np_utils.to_categorical(y_valid, NB_CLASSES)
y_test = np_utils.to_categorical(y_test, NB_CLASSES)


model = DFNet.build(input_shape=INPUT_SHAPE, classes=NB_CLASSES)
model.compile(loss="categorical_crossentropy", optimizer=OPTIMIZER, metrics=["accuracy"])

# Set up Early Stopping callback
callbacks = []
if args.use_early_stopping:
    early_stopping = EarlyStopping(monitor='val_loss', patience=19, restore_best_weights=True)
    callbacks.append(early_stopping)

# Model train 
model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=NB_EPOCH, verbose=2,
          validation_data=(X_valid, y_valid), callbacks=callbacks)

score_test = model.evaluate(X_test, y_test, verbose=2)
print("Testing accuracy:", score_test[1])


print(f"Log saved at: {LOG_FULLPATH}")