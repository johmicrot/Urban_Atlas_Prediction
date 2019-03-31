"""Creator: Daniel Pototzky"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import copy
import collections
from sklearn.metrics import accuracy_score
import math

TOTAL_TRAIN_CSV = '/path/to/Dataframes/train_dataframe.csv'
SUBPART_TRAIN_CSV = '/path/to/Dataframes/train_dataframe_subpart.csv'
ENSEMBLE_PRED = '/path/to/ensemble_prediction_matrix.npy'


def unlabelled_images(total_train, subpart_train):
    # Load total dataframe
    train_53k = pd.read_csv(total_train, sep=',', index_col=0)
    # Load dataframe on which model was pretrained
    train_df_subpart = pd.read_csv(subpart_train, sep=',', index_col=0)
    # Determine unlabelled images
    filenames_53k = train_53k['filename'].tolist()
    filenames_subpart = train_df_subpart['filename'].tolist()
    filenames_unlabelled_images = list(np.setdiff1d(filenames_53k, filenames_subpart))
    train_unlabelled = train_53k[train_53k['filename'].isin(filenames_unlabelled_images)]
    train_unlabelled = train_unlabelled.reset_index(drop=True)
    train_unlabelled = train_unlabelled.rename(index=str, columns={"class1": "class1_true_label"})
    return train_unlabelled


def new_labels_argmax(df, file_ensemble_pred):
    df['class1'] = None
    ensemble_prediction_matrix = np.load(file_ensemble_pred)
    pred = np.argmax(ensemble_prediction_matrix, axis=1)
    df['class1'] = pred
    df = df.drop(['class1_true_label'], axis=1)
    return df


def new_labels_threshold(df_tmp, file_ensemble_pred, threshold):
    df_tmp['class1'] = None
    ensemble_prediction_matrix = np.load(file_ensemble_pred)
    pred = np.argmax(ensemble_prediction_matrix, axis=1)
    df_tmp['pred_certainty'] = np.max(ensemble_prediction_matrix, axis=1)
    pred = np.asarray(list(pred))
    df_tmp['pred_class1']=pred
           
    df_tmp.loc[ df_tmp['pred_certainty']>0.8, 'class1'] = df_tmp['pred_class1']
    df_tmp = df_tmp.dropna()
    df_tmp = df_tmp.reset_index(drop=True)
    df_tmp['class1'] = pd.to_numeric(df_tmp['class1'])
    acc2 = accuracy_score(df_tmp['class1'], df_tmp['class1_true_label'])
    print('accuracy threshold', acc2)
    df_tmp = df_tmp.drop(['class1_true_label', 'pred_certainty', 'pred_class1'], axis=1)
    return df_tmp

       
def new_labels_threshold_and_evening(df_tmp2, file_ensemble_pred, subpart_train, threshold):
    df_tmp2['class1'] = None
    ensemble_prediction_matrix = np.load(file_ensemble_pred)
    pred = np.argmax(ensemble_prediction_matrix, axis=1)
    df_tmp2['pred_certainty'] = np.max(ensemble_prediction_matrix, axis=1)
    pred = np.asarray(list(pred))
    df_tmp2['pred_class1']=pred
    train_subpart = pd.read_csv(subpart_train, sep=',', index_col = 0)
    freq_true_classes=collections.Counter(train_subpart['class1'])

    total = sum(freq_true_classes.values())
    perc_true_classes = {k: v / total for k, v in freq_true_classes.items()}

    perc_actual_classes = {}
    for cl in np.unique(list(perc_true_classes.keys())):
        # get list of pred_certainty for specific class
        perc_actual_classes[cl] = len(df_tmp2.loc[df_tmp2['pred_class1'] == cl]['pred_certainty'])
    total2 = sum(perc_actual_classes.values())
    perc_actual_classes = {k: v / total2 for k, v in perc_actual_classes.items()}
    print('perc_actual_classes', perc_actual_classes)  

    dict_rel_freq = {}
    for cl in np.unique(list(perc_true_classes.keys())):
        dict_rel_freq[cl] = perc_actual_classes[cl]/perc_true_classes[cl]
    print('This is the relative frequency', dict_rel_freq)

    min_rel_freq = min(dict_rel_freq.values())
    print('min_rel_freq', min_rel_freq)

    weight_dict = {}
    
    for cl in np.unique(list(perc_true_classes.keys())):
        weight_dict[cl] = min_rel_freq/dict_rel_freq[cl]
    print('weight_dict', weight_dict)

    # Find threshold for inclusion of individual classes
    for cl in np.unique(list(perc_true_classes.keys())):
        print('len df tmp 2', len(df_tmp2))
        expected_nbimg_of_class = math.floor(weight_dict[cl] * perc_actual_classes[cl] * len(df_tmp2))
        print('expected_nbimg_of_class', expected_nbimg_of_class)
        tmp_cl = df_tmp2.loc[ df_tmp2['pred_class1'] == cl]['pred_certainty']
        
        class_threshold = tmp_cl.sort_values(ascending=False).iloc[expected_nbimg_of_class-1]
        print('class_threshold', class_threshold)
        
        if class_threshold < threshold:
            class_threshold = threshold
        print('class_threshold', class_threshold)
        
        df_tmp2.loc[ (df_tmp2['pred_certainty']>class_threshold)& (df_tmp2['pred_class1'] == cl), 'class1'] = cl
        if cl == 0:
            df_even = df_tmp2.loc[ (df_tmp2['pred_certainty'] > class_threshold)& (df_tmp2['pred_class1'] == cl)]
        else:
            df_even_tmp = df_tmp2.loc[ (df_tmp2['pred_certainty']>class_threshold)& (df_tmp2['pred_class1'] == cl)]
            df_even = df_even.append(df_even_tmp, ignore_index=True)
        
    df_even = df_even.reset_index(drop = True)
    df_even['class1'] = pd.to_numeric(df_even['class1'])
    
    acc_dict = {}
    for cl in np.unique(list(perc_true_classes.keys())):
        acc_dict[cl] = accuracy_score(df_even.loc[df_even['class1'] == cl, 'class1'],
                                      df_even.loc[df_even['class1'] == cl, 'class1_true_label'])
    print('acc dict', acc_dict)
    
    acc2 = accuracy_score(df_even['class1'], df_even['class1_true_label'])
    print('accuracy even', acc2)
    df_even = df_even.drop(['class1_true_label', 'pred_certainty', 'pred_class1'], axis=1)
    return df_even   


# Extract images that are unlabeled
train_unlabelled = unlabelled_images(TOTAL_TRAIN_CSV, SUBPART_TRAIN_CSV)
# Create ensembel labels based on argmax
train_argmax_labels = new_labels_argmax(copy.deepcopy(train_unlabelled), ENSEMBLE_PRED)
# Create ensemble labels based on threshold
train_threshold_labels = new_labels_threshold(copy.deepcopy(train_unlabelled), ENSEMBLE_PRED, 0.8)
# Compute dataframe when using both a threshold and class evening
train_threshold_labels_and_evening = new_labels_threshold_and_evening(copy.deepcopy(train_unlabelled),
                                                                      ENSEMBLE_PRED, SUBPART_TRAIN_CSV, 0.6)


# Save output
train_argmax_labels.to_csv('Dataframes/train_dataframe_ARGMAX_LABELLED_BY_2_5k.csv')

train_threshold_labels.to_csv('Dataframes/train_dataframe_THRESHOLD_LABELLED_BY_2_5k.csv')

train_threshold_labels_and_evening.to_csv('Dataframes/train_dataframe_THRESHOLD_0_6_EVENING_LABELLED_BY_2_5k.csv')

