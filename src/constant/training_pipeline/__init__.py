import os
import sys 
import numpy as np
import pandas as pd

"""
    Defining common constant variable for traning pipeline

"""
TARGET_NAME= "Result"
PIPELINE_NAME: str = "NetworkSecurity"
ARTIFACT_DIR: str = "Artifacts"
FILE_NAME: str = "phisingData.csv"

TRAIN_FILE_NAME:str = "train.csv"
TEST_FILE_NAME: str = "test.csv"



"""
Data Ingesting related variable name
"""
DATA_INGESTION_COLLECTION_NAME:str = "networkdata"
DATA_INGESTION_DATABASE_NAME:str = "networksecurity"
DATA_INGESTION_DIR_NAME:str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_NAME:str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.2 



