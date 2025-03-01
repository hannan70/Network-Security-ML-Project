from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.entity.config_entity import TrainingPipelineConfig, DataIngestionConfig, DataValidationConfig, DataTransformationConfig, ModelTrainerConfig
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.logging.logger import logging

if __name__=="__main__":
    trainingpipelineconfig = TrainingPipelineConfig()
    dataingestionconfig = DataIngestionConfig(trainingpipelineconfig)
    obj = DataIngestion(dataingestionconfig)
    dataingestionartiface = obj.initiate_data_ingestion()
    logging.info("Data ingestion complete")
    # print(dataingestionartiface)

    data_validation_config = DataValidationConfig(trainingpipelineconfig)
    data_validation = DataValidation(data_validation_config, dataingestionartiface)
    logging.info("Initaite the data validation")
    data_validation_artifact = data_validation.initiate_data_validation()
    
    # print("varlidation inside main.py",data_validation_artifact)

    logging.info("Data Transformation start")
    data_transformation_config = DataTransformationConfig(trainingpipelineconfig)
    data_transformation = DataTransformation(data_validation_artifact=data_validation_artifact, data_transformation_config=data_transformation_config)
    data_transformation_artifact = data_transformation.initiate_data_transformation()
    # print(data_transformation_artifact)

    logging.info("Model training start")
    model_trainer_config = ModelTrainerConfig(trainingpipelineconfig)
    modeltrainer = ModelTrainer(model_trainer_config, data_transformation_artifact)
    modeltrainerartiface = modeltrainer.initiate_model_trainer()
    print(modeltrainer)

    


    