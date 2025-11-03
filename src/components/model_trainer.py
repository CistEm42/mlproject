# TRAIN DIFFERENT MODELS AND SEE THE SCORES AND ACCURACY OF EACH MODEL.
import os
import sys
from dataclasses import dataclass
from catboost import CatBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor)
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    
    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting the training and test input data")
            X_train, y_train, X_test, y_test=(
                train_array[:,:-1] , # select all rows, all columns except last column → features,
                train_array[:,-1] ,# select all rows, only last column → target variable,
                test_array[:, :-1],
                test_array[:, -1]
            )
        
            models = {
                "Linear Regression": LinearRegression(),
                "Lasso": Lasso(),
                "Ridge": Ridge(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest Regressor": RandomForestRegressor(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "Ada Boost Regressor": AdaBoostRegressor()
                }

            params = {
                "Linear Regression": {},

                "Lasso": {
                    'alpha': [0.1, 0.5, 1.0]
                },

                "Ridge": {
                    'alpha': [0.1, 0.5, 1.0]
                },

                "K-Neighbors Regressor": {
                    'n_neighbors': [3, 5, 7, 9]
                },

                "Decision Tree": {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    'splitter': ['best', 'random']
                },

                "Random Forest Regressor": {
                    'n_estimators': [8, 16, 32, 64, 128, 256],
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson']
                },

                "XGBRegressor": {
                    'learning_rate': [0.1, 0.01, 0.05, 0.001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },

                "CatBoosting Regressor": {
                    'depth': [6, 8, 10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },

                "Ada Boost Regressor": {
                    'learning_rate': [0.1, 0.01, 0.5, 0.001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                }
            }


            model_report:dict=evaluate_models(X_train=X_train, y_train= y_train,
                                              X_test=X_test, y_test=y_test, models=models, param=params)
            
            ## Getting the best model score from the dictionary
            best_model_score = max(sorted(model_report.values()))

            ## Getting the best model name from the dictionary
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found")
            logging.info("Best model found on both training and testing dataset.")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path, obj=best_model
            )

            predicted=best_model.predict(X_test)

            r2_square = r2_score(y_test,predicted)

            return r2_square

        except Exception as e:
            raise CustomException(e,sys)