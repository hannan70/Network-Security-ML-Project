import os
import sys
import numpy as np
from src.exception.exception import CustomException
from src.logging.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
import pymongo
from dotenv import load_dotenv
load_dotenv()

from src.entity.config_entity import DataIngestionConfig 
from src.entity.artifact_entity import DataIngestionArtifact

MONGO_DB_URL = os.getenv("MONGO_DB_URL")

class DataIngestion:
    def __init__(self, data_ingestion_config:DataIngestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config
        except CustomException as e:
            raise CustomException(e, sys)

    
    # get data form mongodb database
    def export_collection_as_dataframe(self):
        try:
            database_name = self.data_ingestion_config.database_name
            collection_name = self.data_ingestion_config.collection_name

            self.mongo_client = pymongo.MongoClient(MONGO_DB_URL)
            collection = self.mongo_client[database_name][collection_name]

            df = pd.DataFrame(list(collection.find()))
            
            if "_id" in df.columns.to_list():
                df = df.drop(columns=['_id'], axis=1)

            df.replace({"na": np.nan}, inplace=True)
            
            return df
            
        except CustomException as e:
            raise CustomException(e, sys)

    # save raw data inside feature store folder
    def export_data_into_feature_store(self, dataframe:pd.DataFrame):
        try:
            feature_store_file_path = self.data_ingestion_config.feature_store_file_path
            dir_name = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_name, exist_ok=True)
            dataframe.to_csv(feature_store_file_path, index=False, header=True)
            return dataframe

        except CustomException as e:
            raise CustomException(e, sys)
    
    # Spliting dataset into train and test set
    def split_data_as_trian_and_test(self, dataframe:pd.DataFrame):
        try:
            train_set, test_set = train_test_split(dataframe, test_size=self.data_ingestion_config.train_test_split_ration) 

            logging.info("Spliting dataset into train and test set")

            dir_path = os.path.dirname(self.data_ingestion_config.training_file_path) 
            os.makedirs(dir_path, exist_ok=True)

            # save train dataset
            train_set.to_csv(self.data_ingestion_config.training_file_path, index=False, header=True)

            # save test dataset
            test_set.to_csv(self.data_ingestion_config.test_file_path, index=False, header=True)
            logging.info("Successfully exported train and test data ")

        except CustomException as e:
            raise CustomException(e, sys)
    
    # check all function work or not and return train and test path
    def initiate_data_ingestion(self):
        try:
            exp_dataframe = self.export_collection_as_dataframe()
            dataframe = self.export_data_into_feature_store(dataframe=exp_dataframe)
            self.split_data_as_trian_and_test(dataframe)
            
            # return test and train file path
            dataingestionartifact = DataIngestionArtifact(train_file_path=self.data_ingestion_config.training_file_path,
                                                      test_file_path=self.data_ingestion_config.test_file_path)
            
            return dataingestionartifact
        
        except CustomException as e:
            raise CustomException(e, sys)
        