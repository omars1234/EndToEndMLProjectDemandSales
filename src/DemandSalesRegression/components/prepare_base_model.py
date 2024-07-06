import sys
import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder,StandardScaler,OneHotEncoder
from sklearn.ensemble import GradientBoostingRegressor

from DemandSalesRegression import logger
from DemandSalesRegression.utils.common import get_size

from DemandSalesRegression.config.configuration import PrepareBaseModelConfig
import os



class PrepareBaseModel:
    def __init__(self,config:PrepareBaseModelConfig):
        self.config= config
    

    def get_data_transformer_object(self):
        numerical_columns = ['quantity', 'price_per_case', 'total_sales']
        categorical_columns =  ['company_region', 'product','unit']

        num_pipeline= Pipeline(steps=[
            ("scaler",StandardScaler())
            ])
        
        cat_pipeline=Pipeline(steps=[
            ("OneHotEncoder",OneHotEncoder(handle_unknown="ignore")),
            ("scaler",StandardScaler(with_mean=False))
            ])

        preprocessor=ColumnTransformer([
            ("num_pipeline",num_pipeline,numerical_columns),
            ("cat_pipelines",cat_pipeline,categorical_columns)
            ])  
        
        return preprocessor
    
    def initiate_data_transformation(self): 
        train_df=pd.read_csv(r"./artifacts/data_ingestion/train_data.csv")
        test_df=pd.read_csv(r"./artifacts/data_ingestion/test_data.csv")

        logger.info("Read train and test data completed")

        preprocessing_obj=self.get_data_transformer_object()        

        target_column_name='total_sales'
        input_column_name=['quantity', 'price_per_case', 'total_sales','company_region', 'product','unit']       

        input_feature_train_df=train_df[input_column_name]
        target_feature_train_df=train_df[target_column_name]

        input_feature_test_df=test_df[input_column_name]
        target_feature_test_df=test_df[target_column_name]     

        input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
        input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

        train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
                ]
        test_arr = np.c_[
                 input_feature_test_arr, np.array(target_feature_test_df)
                 ]
        
        if not os.path.exists(self.config.train_data_path):
                train_arr = pd.DataFrame(train_arr)#columns=[colnames])
                train_arr.to_csv(self.config.train_data_path,index=False)
        if not os.path.exists(self.config.test_data_path):
                test_arr = pd.DataFrame(test_arr)#,columns=[colnames])
                test_arr.to_csv(self.config.test_data_path,index=False)        

        return(
                self.config.train_data_path,
                self.config.test_data_path
         
        )  