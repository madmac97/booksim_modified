# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 13:19:36 2019

@author: skmandal, anallam1
"""

import os
import math
import shutil
import random
import scipy.io
import numpy as np
import pandas as pd                
#import matplotlib.pyplot as plt
#import sklearn

from functions import *
from sklearn.model_selection import train_test_split  
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression   
from datetime import datetime
from time import sleep

import csv
import random

#for plotting
#from IPython.display import clear_output 
# for Earlystop

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

class ApplicationEnv:
    def __init__(self):

        # Total of columns: 9
        self.col_name_lst = []
        for i in range(1,257):
            self.col_name_lst.append("latency_"+str(i))
        for i in range(1,257):
            self.col_name_lst.append("std_arrival_"+str(i))
        for i in range(1,257):
            self.col_name_lst.append("mean_arrival_"+str(i))
        self.col_name_lst.append("average_latency")

        
        self.col = {k: v for v, k in enumerate(self.col_name_lst)}
       
        self.feature_names = []
        for i in range(1,257):
            self.feature_names.append("std_arrival_"+str(i))
        for i in range(1,257):
            self.feature_names.append("mean_arrival_"+str(i))


        self.f_name = {k: v for v, k in enumerate(self.feature_names)}
        
        self.num_features = len(self.feature_names)
        
        
    def f_train_model(self, feature_data,  labels, classifier_type, max_tree_depth):
        
        X_train, X_test, y_train, y_test = train_test_split(feature_data, labels, test_size=0.20, train_size=0.80, random_state=0)
        
        if classifier_type == 'RT':
            regressor = DecisionTreeRegressor(max_depth = max_tree_depth)
            regressor.fit(X_train, y_train)
        elif classifier_type == 'LR':
            regressor = LinearRegression()
            regressor.fit(X_train, y_train)
            
            #return test accuracy
            #SKM: code is incomplete
#            regressor.predict(X_test)
        elif classifier_type == 'NN':
            regressor = self.f_build_model(self.num_features)
            
            
            regressor.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=500, 
                      epochs=500, verbose=0, shuffle=True) 
        else:
            raise Exception('Unexpected classifier type')
            
        return regressor
       
    def f_test_model(self, feature_data, regressor):
        
        data_length = len(feature_data)
        output_labels = np.zeros(shape=(data_length,1))
        
        for data_idx in range(0, data_length):
            data = feature_data.iloc[[data_idx]]
            output_labels[data_idx] = regressor.predict(data)
            
        return output_labels
    
 
    
    def f_build_model(self, num_features):
        # Neural Net Model
        model = Sequential()
        model.add(Dense(100, input_dim=num_features, kernel_initializer='random_uniform', activation='relu'))      
        #model.add(Dense(50, kernel_initializer='random_uniform', activation='relu'))                                 # Hidden layer
        model.add(Dense(50, kernel_initializer='random_uniform', activation='relu'))                                 # Hidden layer                                # Hidden layer
        model.add(Dense(1, kernel_initializer='random_uniform', activation='linear'))                # Output layer
        
        # Compile the model. Loss function is categorical crossentropy
        model.compile(loss='mse', 
                  optimizer=Adam(lr=0.001), 
                  metrics=['mean_absolute_error'])        
        
        return model
    
       
    def f_plot(self, reference_stress, output_stress):
        
        difference = abs(reference_stress - output_stress)
        
        plt.plot(difference, 'r^-', label="difference")
        
        plt.plot(reference_stress, 'b*-', label="reference_stress")
        plt.show()
                
