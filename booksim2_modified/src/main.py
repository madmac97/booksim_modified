# -*- coding: utf-8 -*-
"""
Created on Thu Sept14 13:18:59 2019

@author: skmandal
"""

import os
import math
import shutil
import random
import scipy.io
import numpy as np
import pandas as pd                
import matplotlib.pyplot as plt
import sklearn
import pickle

from functions import *
from sklearn.model_selection import train_test_split  
from sklearn.tree import DecisionTreeRegressor      
from datetime import datetime
from time import sleep

import csv
import random

import itertools

from keras.models import Sequential, load_model
from keras.layers import Dense
from keras import optimizers
from keras import regularizers
from datetime import datetime
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.callbacks import LearningRateScheduler
from keras import backend as K
from keras.utils.np_utils import to_categorical

classifier_type = "NN" #"NN" #"LR" #"RT"

#max_tree_depth_array = [10, 12, 14, 16, 18, 20, 24, 28, 32, 40, 50]
max_tree_depth_array = [0]

env = ApplicationEnv()

datafile_name = "dataframe.csv"


alldata_orig = pd.read_csv(datafile_name, header = 0)

labels_all = alldata_orig.loc[:, "average_latency"]
feature_data_all = alldata_orig.loc[:, env.feature_names]
#binned data

feature_data_train, feature_data_test, labels_train, labels_test = train_test_split(feature_data_all, labels_all, test_size=0.50, train_size=0.50, random_state=0)


alldata_train = alldata_orig
for max_tree_depth in max_tree_depth_array:

    if classifier_type == "NN":
        filename = classifier_type + "_model_" + str(max_tree_depth) + ".h5"
    else:
        filename = classifier_type + "_model_" + ".sav"
        
    
    phase = "training"
    
    model = env.f_train_model(feature_data_train,  labels_train, classifier_type, max_tree_depth)
    
    if classifier_type == "NN":
        model.save(filename)
    else:
        pickle.dump(model, open(filename, 'wb'))
        
    phase = "testing"

    if classifier_type == "NN":
        regressor = load_model(filename)
    else:
        regressor = pickle.load(open(filename, 'rb'))
    
    
    output_label = env.f_test_model(feature_data_test, regressor)    
    mean_error = np.mean(100*np.abs(labels_test.values.reshape( len(output_label), 1) - output_label)/labels_test.values.reshape( len(output_label), 1))
    
        
#    perc_accuracy = 100*(accuracy/data_length)
#    print("Percentage accuracy for selecting the bin: ", perc_accuracy)
    
    print("Mean absolute percentage error: ", mean_error)
#    print("Mean percentage Error: ", mean_perc_error)
#    print("Mean squared Error: ", mean_sq_error)
    print("")
    