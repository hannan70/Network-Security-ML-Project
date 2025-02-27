import os
import sys
import json
import pymongo

from dotenv import load_dotenv
load_dotenv()

MONGO_DB_URL = os.getenv("MONGO_DB_URL")
print(MONGO_DB_URL)

import certifi
ca = certifi.where()

import numpy as np
import pandas as  pd
from src.exception.exception import CustomException
from src.logging.logger import logging

class NetworkDataExtract:
    try:
        pass
    except CustomException as e:
        raise CustomException(e, sys)


    def cv_to_json_converter(self, file_path):
        try:
            data = pd.read_csv(file_path)
            data.reset_index(drop=True, inplace=True)
            records = list(json.loads(data.T.to_json()).values())
            return records

        except CustomException as e:
            raise CustomException(e, sys)

    def insert_data_mongodb(self, records, database, collection):
        try:
            self.database = database
            self.records = records
            self.collection = collection
            self.mongo_client = pymongo.MongoClient(MONGO_DB_URL)

            self.database = self.mongo_client[self.database]
            self.collection = self.database[self.collection]
            self.collection.insert_many(self.records)

            return len(self.records)

        except CustomException as e:
            raise CustomException(e, sys)
        

if __name__=="__main__":
    file_path = "notebooks\data\phisingData.csv"
    database = "networksecurity"
    collection = "networkdata"
    network_obj = NetworkDataExtract()
    # records = network_obj.cv_to_json_converter(file_path)
    # no_of_records = network_obj.insert_data_mongodb(records, database, collection)
    # print(no_of_records)
    # print(records)
