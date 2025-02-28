from datetime import datetime
import os
from src.constant import training_pipeline

print(training_pipeline.ARTIFACT_DIR)
print(training_pipeline.PIPELINE_NAME)


class TrainingPipelineConfig:
    def __init__(self,timestamp=datetime.now()):
        timestamp=timestamp.strftime("%m_%d_%Y_%H_%M_%S")
        self.pipeline_name=training_pipeline.PIPELINE_NAME
        self.artifact_name=training_pipeline.ARTIFACT_DIR
        # example Artifacts\02_28_2025_14_29_10
        self.artifact_dir=os.path.join(self.artifact_name,timestamp)
        self.timestamp: str=timestamp
        


class DataIngestionConfig:
    def __init__(self, train_pipeline_config:TrainingPipelineConfig):
        # example: Artifacts\02_28_2025_14_32_29\data_ingestion
        self.data_ingestion_dir:str = os.path.join(
            train_pipeline_config.artifact_dir, training_pipeline.DATA_INGESTION_DIR_NAME
        )

        # Example: Artifacts\02_28_2025_14_36_25\data_ingestion\feature_store\phisingData.csv
        self.feature_store_file_path:str = os.path.join(
            self.data_ingestion_dir, training_pipeline.DATA_INGESTION_FEATURE_STORE_NAME, training_pipeline.FILE_NAME
        )

        # Example: Artifacts\02_28_2025_14_39_22\data_ingestion\ingested\train.csv
        self.training_file_path = os.path.join(
            self.data_ingestion_dir, training_pipeline.DATA_INGESTION_INGESTED_DIR, training_pipeline.TRAIN_FILE_NAME 
        )

        # Example: Artifacts\02_28_2025_14_41_23\data_ingestion\ingested\test.csv
        self.test_file_path = os.path.join(
            self.data_ingestion_dir, training_pipeline.DATA_INGESTION_INGESTED_DIR, training_pipeline.TEST_FILE_NAME
        )

        self.train_test_split_ration:float = training_pipeline.DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO
        self.collection_name: str = training_pipeline.DATA_INGESTION_COLLECTION_NAME
        self.database_name: str = training_pipeline.DATA_INGESTION_DATABASE_NAME
       


 


