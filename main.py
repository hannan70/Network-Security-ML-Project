from src.components.data_ingestion import DataIngestion
from src.entity.config_entity import TrainingPipelineConfig, DataIngestionConfig
from src.logging.logger import logging

if __name__=="__main__":
    trainingpipelineconfig = TrainingPipelineConfig()
    dataingestionconfig = DataIngestionConfig(trainingpipelineconfig)
    obj = DataIngestion(dataingestionconfig)
    dataingestionartiface = obj.initiate_data_ingestion()
    logging.info("Data ingestion complete")
    print(dataingestionartiface)
    