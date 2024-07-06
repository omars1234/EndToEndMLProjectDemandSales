
from DemandSalesRegression.config.configuration import ConfigurationManager
from DemandSalesRegression.components.prepare_base_model import PrepareBaseModel
from DemandSalesRegression.utils.common import logger


STAGE_NAME = "prepare base model stage"

class PrepareBaseModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        prepare_base_model_config = config.get_prepare_base_model_config()
        prepare_base_model = PrepareBaseModel(prepare_base_model_config)
        prepare_base_model.get_data_transformer_object()
        prepare_base_model.initiate_data_transformation()



if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = PrepareBaseModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
        