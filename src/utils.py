#all the common functionalites that the entire project needs
import os
import sys #to handle exceptions

import numpy as np 
import pandas as pd
import dill # to create pickle files
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exceptions import CustomException

def save_object(file_path, obj):
    try:
        #  Get the directory path of the file to ensure it exists before saving
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)
         # Save the object as a pickle file
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train,X_test,y_test,models,param):
    try:
        report = {}
        # Perform GridSearchCV for hyperparameter tuning
        for i in range(len(list(models))):
            model = list(models.values())[i]
            para=param[list(models.keys())[i]]

            # Perform GridSearchCV for hyperparameter tuning
            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train,y_train)


            # Set the best parameters found from GridSearchCV
            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)

            #model.fit(X_train, y_train)  # Train model

            # Predict on both training and test data
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Calculate R2 score for both training and test predictions
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            # Add the test model score to the report dictionary
            report[list(models.keys())[i]] = test_model_score
        # Return the report with model names as keys and their R2 scores as value    
        return report

    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        # Load and return the object saved in the pickle file
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)