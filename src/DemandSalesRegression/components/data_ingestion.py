import os
import urllib.request as request
from DemandSalesRegression import logger
from DemandSalesRegression.utils.common import get_size
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.pipeline import Pipeline
import pandas as pd

from DemandSalesRegression.entity.config_entity import DataIngestionConfig



class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config
        


    def download_file(self):
        if not os.path.exists(self.config.local_data_file):
            df=pd.read_csv(self.config.source_URL)
            df.to_csv(self.config.local_data_file,index=False,header=True)

            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)

            if not os.path.exists(self.config.train_data_path):
                train_set.to_csv(self.config.train_data_path,index=False,header=True)            
        
            if not os.path.exists(self.config.test_data_path):
                test_set.to_csv(self.config.test_data_path,index=False,header=True) 

            return(
                self.config.train_data_path,
                self.config.test_data_path
            )     


        logger.info("Inmgestion of the data is completed")
       
        