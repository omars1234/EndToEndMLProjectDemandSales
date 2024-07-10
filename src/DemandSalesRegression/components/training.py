import os
import urllib.request as request
from sklearn.ensemble import GradientBoostingRegressor
import time
from sklearn.metrics import r2_score
from sklearn.metrics import r2_score,mean_absolute_percentage_error
import pickle
import pandas as pd
from numpy import loadtxt

from DemandSalesRegression.config.configuration import TrainingConfig

class Training:
    def __init__(self,config:TrainingConfig):
        self.config= config

    def initiate_Training(self):   
        
        input_feature_train_arr=loadtxt('artifacts/prepare_base_model/train_data', delimiter=',')
        input_feature_test_arr=loadtxt('artifacts/prepare_base_model/test_data', delimiter=',')
        x_train,y_train,x_test,y_test=(
            input_feature_train_arr[:,:-1],
            input_feature_train_arr[:,-1],
            input_feature_test_arr[:,:-1],
            input_feature_test_arr[:,-1]
        )
        model=GradientBoostingRegressor(
                subsample=self.config.subsample,
                n_estimators=self.config.n_estimators,
                min_samples_split=self.config.min_samples_split,
                min_samples_leaf=self.config.min_samples_leaf,
                max_depth=self.config.max_depth,
                learning_rate=self.config.learning_rate
        )
        model.fit(x_train,y_train)

        import pickle
        with open ("./artifacts/final_model.pkl","wb") as f:
           pickle.dump(model,f)