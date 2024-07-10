
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: Path
    local_data_file: Path
    train_data_path: Path
    test_data_path: Path



@dataclass(frozen=True)
class PrepareBaseModelConfig:
    root_dir: Path
    base_model_path:Path
    #updated_base_model_path:Path
    train_data_path: Path
    test_data_path: Path


@dataclass(frozen=True)
class TrainingConfig:
    root_dir:Path
    trained_model_file_path:Path
    updated_base_model_path:Path
    training_data:Path
    testing_data:Path
    subsample:float
    n_estimators:int
    min_samples_split:int
    min_samples_leaf:int
    max_depth:int
    learning_rate:float


@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir:Path
    model_path:Path
    all_params:dict
    metrix_file_name:Path
    training_data:Path
    testing_data:Path
    #mlflow_uri: str       
