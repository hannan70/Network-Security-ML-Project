from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.entity.config_entity import (
    TrainingPipelineConfig, 
    DataIngestionConfig, 
    DataValidationConfig, 
    DataTransformationConfig, 
    ModelTrainerConfig
)
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
    DataTransformationArtifact,
    ModelTrainerArtifact
)
from src.logging.logger import logging
from src.exception.exception import CustomException
import sys

class TrainingPipeline:
    def __init__(self):
        self.training_pipeline_config = TrainingPipelineConfig()


    def start_data_ingestion(self):
        try:
            self.data_ingestion_config = DataIngestionConfig(train_pipeline_config=self.training_pipeline_config)
            logging.info("Start data ingestion")
            data_ingestion = DataIngestion(data_ingestion_config=self.data_ingestion_config)
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            logging.info(f"Data ingestion completed and artifact {data_ingestion_artifact}")
            return data_ingestion_artifact

        except CustomException as e:
            raise CustomException(e, sys)
        

    def start_data_validation(self, data_ingestion_artifact:DataIngestionArtifact):
        try:
            data_validation_config = DataValidationConfig(training_pipeline_config=self.training_pipeline_config)
            data_validation = DataValidation(data_validation_config, data_ingestion_artifact=data_ingestion_artifact)
            logging.info("Initaite the data validation")
            data_validation_artifact = data_validation.initiate_data_validation()
            return data_validation_artifact
        except CustomException as e:
            raise CustomException(e, sys)
        
    def start_data_transformation(self, data_validation_artifact:DataValidationArtifact):
        try:
            logging.info("Data Transformation start")
            data_transformation_config = DataTransformationConfig(training_pipeline_config=self.training_pipeline_config)
            data_transformation = DataTransformation(data_validation_artifact=data_validation_artifact, data_transformation_config=data_transformation_config)
            data_transformation_artifact = data_transformation.initiate_data_transformation()
            return data_transformation_artifact

        except CustomException as e:
            raise CustomException(e, sys)
        

    def start_model_trainer(self, data_transformation_artifact:DataTransformationArtifact):
        try:
            logging.info("Model training start")
            model_trainer_config = ModelTrainerConfig(training_pipeline_config=self.training_pipeline_config)
            model_trainer = ModelTrainer(model_trainer_config, data_transformation_artifact)
            model_trainer_artifact = model_trainer.initiate_model_trainer()
            return model_trainer_artifact

        except CustomException as e:
            raise CustomException(e, sys)
        

    def run_pipeline(self):
        try:
            data_ingestion_artifact = self.start_data_ingestion()
            data_validation_artifact = self.start_data_validation(data_ingestion_artifact=data_ingestion_artifact)
            data_transformation_artifact = self.start_data_transformation(data_validation_artifact=data_validation_artifact)
            model_trainer_artifact = self.start_model_trainer(data_transformation_artifact=data_transformation_artifact)

            return model_trainer_artifact
        
        except CustomException as e:
            raise CustomException(e, sys)