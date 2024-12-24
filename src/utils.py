import dill  # for handling complex objects
import os
import sys
import logging
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from src.exceptions import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        
        logging.info(f"Creating directory path: {dir_path}")
        os.makedirs(dir_path, exist_ok=True)  # Ensure directory exists

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)  # Using dill for complex objects

        logging.info(f"Model saved to {file_path}")

    except Exception as e:
        logging.error(f"Error occurred while saving the model: {str(e)}")
        raise CustomException(e, sys)


# Evaluate multiple models
def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}
        for i in range(len(list(models))):
            model = list(models.values())[i]
            para = param[list(models.keys())[i]]

            logging.info(f"Evaluating {list(models.keys())[i]}...")

            gs = GridSearchCV(model, para, cv=3, verbose=1)  # Added verbosity
            gs.fit(X_train, y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score
            logging.info(f"{list(models.keys())[i]} Test R2: {test_model_score}")

        return report

    except Exception as e:
        raise CustomException(f"Error while evaluating models: {str(e)}", sys)

# Load model object
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)  # Using dill instead of pickle

    except Exception as e:
        raise CustomException(f"Error while loading object: {str(e)}", sys)
