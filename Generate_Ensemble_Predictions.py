#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Creator: Daniel Pototzky,
   Authors: Daniel Pototzky, John Rothman"""

import pandas as pd
import numpy as np
from keras.models import load_model
import DataGenerator as Dg

# Define path where images are stores
IMG_SOURCE = 'Dataset/'
BATCH_S = 32
# The name of the csv dataset file
DATASET = '2_5k'
# Options are "single_task" and "multi_task"
TASK = 'single_task'

# Can be 'pop_pred' or 'land_type_pred' in our case
TASK_NAME = 'land_type_pred'

# Load total dataframe
train_df_53k = pd.read_csv('Dataframes/train_dataframe.csv', sep=',', index_col = 0)

# Load dataframe on which model was pretrained
train_df_subpart = pd.read_csv('Dataframes/train_dataframe_%s.csv' % DATASET, sep=',', index_col=0)

# Determine unlabelled images
filenames_53k = train_df_53k['filename'].tolist()
filenames_subpart = train_df_subpart['filename'].tolist()
filenames_unlabelled_images = list(np.setdiff1d(filenames_53k,filenames_subpart))
train_df_unlabeled = train_df_53k[train_df_53k['filename'].isin(filenames_unlabelled_images)]
train_df_unlabeled = train_df_unlabeled .reset_index(drop=True)

# Load pretrained model
pretrained_model = load_model('path/to/model.hdf5')

N_CLASSES = len(train_df_unlabeled[TASK_NAME].unique())

# Define Parameters for augmentation
params_default = {'dim': (224, 224, 3),
                  'batch_size': 1,
                  'n_classes1': N_CLASSES,
                  'n_channels': 3,
                  'shuffle': False,
                  'mode': TASK,
                  'class_name': TASK_NAME
                  }


params_flip1 = {'dim': (224, 224, 3),
                'batch_size': 1,
                'n_classes1': N_CLASSES,
                'n_channels': 3,
                'shuffle': False,
                'flip1':True,
                'mode': TASK,
                'class_name': TASK_NAME
                }


params_flip2 = {'dim': (224, 224, 3),
                'batch_size': 1,
                'n_classes1': N_CLASSES,
                'n_channels': 3,
                'shuffle': False,
                'flip2': True,
                'mode': TASK,
                'class_name': TASK_NAME
                }


params_dorotate = {'dim': (224, 224, 3),
                   'batch_size': 1,
                   'n_classes1': N_CLASSES,
                   'n_channels': 3,
                   'shuffle': False,
                   'dorotate': True,
                   'mode': TASK,
                   'class_name': TASK_NAME
                   }

params_dozoom = {'dim': (224, 224, 3),
                 'batch_size': 1,
                 'n_classes1': N_CLASSES,
                 'n_channels': 3,
                 'shuffle': False,
                 'dozoom': True,
                 'mode': TASK,
                 'class_name': TASK_NAME
                 }

list_of_transformations = [params_default, params_flip1, params_flip2, params_dorotate]

ensemble_prediction_mat = np.zeros(len(train_df_unlabeled) * N_CLASSES).reshape(len(train_df_unlabeled), N_CLASSES)

for params in list_of_transformations:
    gen_test = Dg.DataGenerator(train_df_unlabeled, IMG_SOURCE, **params)
    predict = pretrained_model.predict_generator(gen_test, steps=len(train_df_unlabeled))
    ensemble_prediction_mat += predict
    pred_argmax = list(np.argmax(predict, axis=1))
       
    true = 0
    total = 0
    for i in range(len(pred_argmax)):
        total += 1
        if pred_argmax[i] == train_df_unlabeled.loc[i, 'class1']:
            true += 1           
    print(params, 'accuracy', true/total)
    
ensemble_prediction_mat = ensemble_prediction_mat / len(list_of_transformations)

# np.save('/path/to/ensemble_prediction_matrix.npy', ensemble_prediction_matrix)






