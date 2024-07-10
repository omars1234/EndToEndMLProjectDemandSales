
import pickle
import pandas as pd
import numpy as np
from numpy import loadtxt
from pathlib import Path

import urllib.request as request
from urllib.parse import urlparse
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score,mean_squared_error
#import mlflow
#import mlflow.sklearn
import time
from DemandSalesRegression.utils.common import save_json
from DemandSalesRegression.config.configuration import ModelEvaluationConfig


class ModelEvaluation:
    def __init__(self,config:ModelEvaluationConfig):
        self.config= config

    def evaluation_metrics(self):
        with open("./artifacts/final_model.pkl","rb") as f:
           loaded_model=pickle.load(f)
        
        test_data=loadtxt('artifacts/prepare_base_model/test_data', delimiter=',')
        x_test,y_test=(
            
            test_data[:,:-1],
            test_data[:,-1]
            )      
        rmse=np.sqrt(mean_squared_error(y_test,loaded_model.predict(x_test)))
        mse=mean_squared_error(y_test,loaded_model.predict(x_test))
        r2score=r2_score(y_test,loaded_model.predict(x_test))
        Scores={"rmse":rmse,"mse":mse,"r2score":r2score}
        save_json(path=Path("scores.json"),data=Scores)
        #return rmse,mse,r2score
    
"""
    def log_into_mlflow(self):

        with open("./artifacts/final_model.pkl","rb") as f:
           loaded_model=pickle.load(f)
        
        test_data=loadtxt('artifacts/prepare_base_model/test_data', delimiter=',')
        x_test,y_test=(
            
            test_data[:,:-1],
            test_data[:,-1]
            )
        

        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        
        with mlflow.start_run():
            predicted=loaded_model.predict(x_test)
            (rmse,mse,r2score)=self.evaluation_metrics(y_test,predicted)

            Scores={"rmse":rmse,"mse":mse,"r2score":r2score}
            save_json(path=Path("scores.json"),data=Scores)
            
        
            mlflow.log_metrics(
                {"rmse": rmse,"mse":mse ,"r2score": r2score}
                )
            mlflow.log_params(self.config.all_params)


            if tracking_url_type_store != "file":

                mlflow.sklearn.log_model(loaded_model, "model", registered_model_name="Price Regression")
            else:
                mlflow.sklearn.log_model(loaded_model, "model") 
"""              