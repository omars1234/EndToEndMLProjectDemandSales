
from DemandSalesRegression.constants import *
from DemandSalesRegression.utils.common import read_yaml,create_directories
from DemandSalesRegression.entity.config_entity import (DataIngestionConfig,PrepareBaseModelConfig,
                                                        TrainingConfig,ModelEvaluationConfig)
import os

class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):
        #schema_filepath=SCHEMA_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        #self.schema=read_yaml(schema_filepath)

        create_directories([self.config.artifacts_root])


    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        #schema=self.schema.COLUMNS

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            train_data_path= config.train_data_path,
            test_data_path=config.test_data_path
        )

        return data_ingestion_config
    


    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        config = self.config.prepare_base_model
        
        create_directories([config.root_dir])

        prepare_base_model_config = PrepareBaseModelConfig(
            root_dir=Path(config.root_dir),
            base_model_path=Path(config.base_model_path),
            #updated_base_model_path=Path(config.updated_base_model_path),
            train_data_path= config.train_data_path,
            test_data_path=config.test_data_path        
            
        )

        return prepare_base_model_config
    

    def get_training_config(self) -> TrainingConfig:
        training = self.config.training
        params=self.params.GradientBoostingRegressor
        prepare_base_model=self.config.prepare_base_model
        training_data=os.path.join(self.config.data_ingestion.train_data_path,"train_data.csv")
        testing_data=os.path.join(self.config.data_ingestion.test_data_path,"test_data.csv")
        create_directories([training.root_dir])

        training_config = TrainingConfig(
            root_dir=Path(training.root_dir),
            trained_model_file_path=Path(training.trained_model_file_path),
            updated_base_model_path=Path(prepare_base_model.updated_base_model_path),
            training_data=Path(training_data),
            testing_data=Path(testing_data),
            subsample=params.SUBSAMPLE,
            n_estimators=params.N_ESTIMATORS,
            min_samples_split=params.MIN_SAMPLES_SPLIT,
            min_samples_leaf=params.MIN_SAMPLES_LEAF,
            max_depth=params.MAX_DEPTH,
            learning_rate=params.LEARNING_RATE
    
        )

        return training_config
    

    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        config = self.config.model_evaluation
        params=self.params.GradientBoostingRegressor

        create_directories([config.root_dir])

        model_evaluation_config = ModelEvaluationConfig(
            root_dir=Path(config.root_dir),
            model_path=Path(config.model_path),
            all_params=params,
            metrix_file_name=Path(config.metrix_file_name),
            training_data=Path(config.training_data),
            testing_data=Path(config.testing_data),
            #mlflow_uri="https://dagshub.com/omars1234/Regression_Analysis.mlflow" 
        )

        return model_evaluation_config