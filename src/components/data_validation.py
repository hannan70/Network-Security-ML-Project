import os, sys
import pandas as pd
from src.entity.config_entity import DataValidationConfig
from src.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from src.exception.exception import CustomException
from src.constant.training_pipeline import SCHEMA_FILE_PATH
from src.utils.utils import read_yaml_file, write_yaml_file
from src.logging.logger import logging
from scipy.stats import ks_2samp

class DataValidation:
    def __init__(self, data_validation_config: DataValidationConfig,
                 data_ingestion_artifact: DataIngestionArtifact):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)
        except CustomException as e:
            raise CustomException(e, sys)
        
    @staticmethod
    def read_data(file_path)->pd.DataFrame:
        try:
           return pd.read_csv(file_path)
        except CustomException as e:
            raise CustomException(e, sys)
    
    # Validate dataframe number of column equal or not
    def validate_number_of_columns(self, dataframe:pd.DataFrame)->bool:
        try:
            number_of_columns= len(self._schema_config['columns'])
            logging.info(f"Required number of columns:{number_of_columns}")
            logging.info(f"Data frame has columns:{len(dataframe.columns)}")
            if len(dataframe.columns) == number_of_columns:
                return True
            return False
        except CustomException as e:
            raise CustomException(e, sys)

    
    # check data distribution
    def detect_dataset_drift(self,train_df,test_df,threshold=0.05)->bool:
        try:
            status=True
            report={}
            for column in train_df.columns:
                d1=train_df[column]
                d2=test_df[column]
                is_same_dist=ks_2samp(d1,d2) # this function use for check data distribution
                if threshold<=is_same_dist.pvalue:
                    is_found=False
                else:
                    is_found=True
                    status=False
                report.update({column:{
                    "p_value":float(is_same_dist.pvalue),
                    "drift_status":is_found
                    
                    }})
            drift_report_file_path = self.data_validation_config.drift_report_file_path

            #Create directory
            dir_path = os.path.dirname(drift_report_file_path)
            os.makedirs(dir_path,exist_ok=True)
            write_yaml_file(file_path=drift_report_file_path,content=report)

        except CustomException as e:
            raise CustomException(e,sys)
          

    def initiate_data_validation(self):
        try:
            train_file_path = self.data_ingestion_artifact.train_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path

            train_dataframe = DataValidation.read_data(train_file_path)
            test_dataframe = DataValidation.read_data(test_file_path)

            # Check if DataFrames are None before validation
            

            if train_dataframe is not None:
                status = self.validate_number_of_columns(dataframe=train_dataframe)
                if not status:
                    logging.info("Train dataframe does not contain all columns.")
                else:
                    logging.info("Train dataframe are equal ")
            else:
                logging.error(f"Failed to load train data from {train_file_path}")

            # check for test dataset
            if test_dataframe is not None:
                status = self.validate_number_of_columns(dataframe=test_dataframe)
                if not status:
                    logging.info("Test dataframe does not contain all columns")
                else:
                    logging.info("Test dataframe are equal ")
            else:
                logging.error(f"Failed to load test data from {test_file_path}")
            
            # data distribution check
            status = self.detect_dataset_drift(train_df=train_dataframe, test_df=test_dataframe)
            dir_path=os.path.dirname(self.data_validation_config.valid_train_file_path)
            os.makedirs(dir_path,exist_ok=True)

            train_dataframe.to_csv(
                self.data_validation_config.valid_train_file_path, index=False, header=True
            )

            test_dataframe.to_csv(
                self.data_validation_config.valid_test_file_path, index=False, header=True
            )
            
            data_validation_artifact = DataValidationArtifact(
                validation_status=status,
                valid_train_file_path=self.data_ingestion_artifact.train_file_path,
                valid_test_file_path=self.data_ingestion_artifact.test_file_path,
                invalid_train_file_path=None,
                invalid_test_file_path=None,
                drift_report_file_path=self.data_validation_config.drift_report_file_path,
            )
            
            return data_validation_artifact

        except CustomException as e:
            raise CustomException(e, sys)