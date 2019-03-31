# -*- coding: utf-8 -*-
"""Authors: Daniel Pototsky, John Rothman"""
import os
import time
from keras.applications.mobilenet import MobileNet
from keras.layers.core import Activation
from keras.layers import Reshape
from keras.models import Model
from keras.layers.convolutional import Conv2D
from keras.callbacks import ModelCheckpoint
from sklearn.utils import class_weight
import pandas as pd
import numpy as np
import DataGenerator as DG


IMG_SOURCE = 'Dataset/'
BATCH_S = 32
#the name of the csv dataset file
DATASET = '2_5k'
#options are "single_task" and "multi_task"
TASK = 'single_task'


# for single task can be 'pop_pred' or 'land_type_pred' in our case
CLASS_NAME_ST = 'land_type_pred'
# for multi task
CLASS_NAMES_MT = ['land_type_pred', 'pop_pred']
LOSS_WEIGHT_CLASS1_MT = 1.
LOSS_WEIGHT_CLASS2_MT = 0.6

train_df = pd.read_csv('Dataframes/train_dataframe_%s.csv' % DATASET, sep=',', index_col=0)
test_dataframe = pd.read_csv('Dataframes/test_dataframe.csv', sep=',', index_col=0)
train_df = train_df.sample(frac=1).reset_index(drop=True)
num_classes_st = len(train_df[CLASS_NAME_ST].unique())
num_classes_mt1 = len(train_df[CLASS_NAMES_MT[0]].unique())
num_classes_mt2 = len(train_df[CLASS_NAMES_MT[1]].unique())
base_model = MobileNet(weights=None, include_top=True)

# Get rid of the classification layer
base_model.layers.pop()
base_model.layers.pop()
base_model.layers.pop()

if TASK == 'single_task':
    #Create new single task layer at the end
    new_conv_preds = Conv2D(num_classes_st, kernel_size=1)(base_model.layers[-1].output)
    new_act_softmax = Activation(activation='softmax')(new_conv_preds)
    newoutput = Reshape((num_classes_st,), name=CLASS_NAME_ST)(new_act_softmax)
    model=Model(input=base_model.input, output=newoutput)

    class_weights = class_weight.compute_class_weight('balanced',
                                                      np.unique(train_df[CLASS_NAME_ST]),
                                                      train_df[CLASS_NAME_ST])
    class_name = CLASS_NAME_ST
    num_classes1 = num_classes_st
    num_classes2 = None
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='Adam', metrics=['accuracy'])

elif TASK == 'multi_task':
    # TASK 1
    new_conv_preds1 = Conv2D(num_classes_mt1, kernel_size=1)(base_model.layers[-1].output)
    new_act_softmax1 = Activation(activation='softmax')(new_conv_preds1)
    newoutput1 = Reshape((num_classes_mt1,), name='reshape_%s' % CLASS_NAMES_MT[0])(new_act_softmax1)
    # TASK 2
    new_conv_preds2 = Conv2D(num_classes_mt2, kernel_size=1)(base_model.layers[-1].output)
    new_act_softmax2 = Activation(activation='softmax')(new_conv_preds2)
    newoutput2 = Reshape((num_classes_mt2,), name='reshape_%s' % CLASS_NAMES_MT[1])(new_act_softmax2)
    loss_weight = [LOSS_WEIGHT_CLASS1_MT, LOSS_WEIGHT_CLASS2_MT]
    model = Model(input=base_model.input, output=[newoutput1, newoutput2])

    # Compute class weights for land class
    names1 = np.unique(train_df[CLASS_NAMES_MT[0]])
    class_weights_MT1 = class_weight.compute_class_weight('balanced',
                                                          names1,
                                                          train_df[CLASS_NAMES_MT[0]])
    names2 = np.unique(train_df[CLASS_NAMES_MT[1]])
    class_weights_MT2 = class_weight.compute_class_weight('balanced',
                                                          names2,
                                                          train_df[CLASS_NAMES_MT[1]])
    class_weights = {}
    for i in range(len(names1)):
        class_weights[names1[i]] = class_weights_MT1[i]
    for i in range(len(names2)):
        class_weights[names2[i]] = class_weights_MT2[i]

    num_classes1 = num_classes_mt1
    num_classes2= num_classes_mt2
    class_name = CLASS_NAMES_MT
    # model.compile(loss='sparse_categorical_crossentropy',
    #               optimizer='Adam', metrics=['accuracy'], loss_weights=loss_weight)
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='Adam', metrics=['accuracy'])
# compile_model(TASK)


model.summary()
save_dir = 'models_Europe/'
folder_name = '%s_%s_b%s_' % (DATASET, CLASS_NAME_ST, BATCH_S) + time.strftime("%Y-%m-%d-%H-%M/", time.gmtime())
save_dir = save_dir + folder_name
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


# class_weights = compute_class_weights(TASK,CLASS_NAME_ST,CLASS_NAMES_MT)
# # for population weights
# name_classes = np.unique(train_df[CLASS_NAME_ST])
# class_weights = class_weight.compute_class_weight('balanced',
#                                                    name_classes,
#                                                    train_df[CLASS_NAME_ST])
#
#

params_train = {'dim': (224, 224, 3),
                'batch_size': BATCH_S,
                'n_classes1': num_classes1,
                'n_classes2': num_classes2,
                'n_channels': 3,
                'shuffle': True,
                'istrain': True,
                'mode': TASK,
                'class_name': class_name}

params_test = {'dim': (224, 224, 3),
               'batch_size': BATCH_S,
               'n_classes1': num_classes1,
               'n_classes2': num_classes2,
               'n_channels': 3,
               'shuffle': False,
               'istrain': False,
               'mode': TASK,
               'class_name': class_name}


gen_train = DG.DataGenerator(train_df, IMG_SOURCE, **params_train)
gen_test = DG.DataGenerator(test_dataframe, IMG_SOURCE, **params_test)



history = model.fit_generator(
    generator=gen_train,
    callbacks=[ModelCheckpoint(save_dir + 'iteration_{epoch:02d}-val_acc_{val_acc:.4f}.hdf5', monitor='val_acc',
                               verbose=1, save_best_only=False)],
    class_weight=class_weights,
    initial_epoch=18,
    max_q_size=1,
    use_multiprocessing=False,
    nb_epoch=22,
    verbose=1,
    validation_data=gen_test)

