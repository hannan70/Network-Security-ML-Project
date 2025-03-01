import os, sys
import pandas as pd
import numpy as np
from src.exception.exception import CustomException
from src.logging.logger import logging
from src.entity.config_entity import ModelTrainerConfig
from src.entity.artifact_entity import DataTransformationArtifact, ClassificationMetricArtifact, ModelTrainerArtifact

from src.utils.utils import save_object, load_object
from src.utils.utils import load_numpy_array_data, evaluate_models
from src.utils.mlutils.metric.classification_metric import get_classification_score

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)


class ModelTrainer:
    def __init__(self,model_trainer_config:ModelTrainerConfig,
                data_transformation_artifact:DataTransformationArtifact):
        try:
            self.model_trainer_config=model_trainer_config
            self.data_transformation_artifact=data_transformation_artifact
        except Exception as e:
            raise CustomException(e,sys)
        

    def train_model(self, X_train, X_test, y_train, y_test):
        models = {
                "Random Forest": RandomForestClassifier(verbose=1),
                "Decision Tree": DecisionTreeClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(verbose=1),
                "Logistic Regression": LogisticRegression(verbose=1),
                "AdaBoost": AdaBoostClassifier(),
            }
        params={
            "Decision Tree": {
                'criterion':['gini', 'entropy', 'log_loss'],
                # 'splitter':['best','random'],
                # 'max_features':['sqrt','log2'],
            },
            "Random Forest":{
                # 'criterion':['gini', 'entropy', 'log_loss'],
                
                # 'max_features':['sqrt','log2',None],
                'n_estimators': [8,16,32,128,256]
            },
            "Gradient Boosting":{
                # 'loss':['log_loss', 'exponential'],
                'learning_rate':[.1,.01,.05,.001],
                'subsample':[0.6,0.7,0.75,0.85,0.9],
                # 'criterion':['squared_error', 'friedman_mse'],
                # 'max_features':['auto','sqrt','log2'],
                'n_estimators': [8,16,32,64,128,256]
            },
            "Logistic Regression":{},
            "AdaBoost":{
                'learning_rate':[.1,.01,.001],
                'n_estimators': [8,16,32,64,128,256]
            }
            
        }
        model_report:dict = evaluate_models(X_train, X_test, y_train, y_test, models, param=params)
         # get best model score from dict
        best_model_score = max(sorted(model_report.values()))
        print(best_model_score)

        for name, score in model_report.items():
                if best_model_score == score:
                    best_model_name = name 

        # like `LinearRegression()` this is the best model for this dataset 
        best_model = models[best_model_name]

        # print(model_report)
        print(best_model)

        # check train classification metrics
        y_train_pred = best_model.predict(X_train)
        classification_train_metric=get_classification_score(y_true=y_train, y_pred=y_train_pred)

        print("train classification score: ", classification_train_metric)

        # check test classification metrics
        y_test_pred = best_model.predict(X_test)
        classification_test_metric = get_classification_score(y_true=y_test, y_pred=y_test_pred)
        print("test classification score: ", classification_test_metric)

        # preprocessor = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)
            
        model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
        os.makedirs(model_dir_path, exist_ok=True)

        save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
        
        #model pusher
        save_object("final_model/model.pkl",best_model)

        ## Model Trainer Artifact
        model_trainer_artifact=ModelTrainerArtifact(trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                             train_metric_artifact=classification_train_metric,
                             test_metric_artifact=classification_test_metric
                             )
        logging.info(f"Model trainer artifact: {model_trainer_artifact}")
        
        return model_trainer_artifact      


    
    def initiate_model_trainer(self)->ModelTrainerArtifact:
        try:
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path

            #loading training array and testing array
            train_arr = load_numpy_array_data(train_file_path)
            test_arr = load_numpy_array_data(test_file_path)

            logging.info("Split training and test input data")
            X_train, X_test, y_train, y_test = (
                train_arr[:, :-1], # X_train
                test_arr[:,:-1], # X_test
                train_arr[:, -1], # y_train
                test_arr[:, -1] # y_test
                
            )


            model_trainer_artifact=self.train_model(X_train,X_test,y_train,y_test)

            return model_trainer_artifact

            
        except Exception as e:
            raise CustomException(e,sys)

