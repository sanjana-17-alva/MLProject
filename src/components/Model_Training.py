import os
import sys
import pickle
import logging
import numpy as np
from sklearn.metrics import r2_score
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from src.exceptions import CustomException  # Custom exception handling from your existing code


# Function to generate detailed error message
def error_message_detail(error, error_detail: sys):
    """
    This function generates a detailed error message with filename, line number, and error message.
    """
    _, _, exc_tb = error_detail.exc_info()  # Get exception details (type, value, and traceback)
    file_name = exc_tb.tb_frame.f_code.co_filename  # Get the file name where the error occurred
    error_message = "Error occurred in Python script [{0}] at line number [{1}] with error message [{2}]".format(
        file_name, exc_tb.tb_lineno, str(error)  # Format the error message
    )
    return error_message  # Return the formatted error message


# Custom exception class that will be raised with detailed error information
class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        """
        Initializes the CustomException with the provided error message and details.
        """
        super().__init__(error_message)  # Initialize base Exception class
        self.error_message = error_message_detail(error_message, error_detail=error_detail)  # Detailed error

    def __str__(self):
        """
        This method is called when the exception is printed or logged. 
        It returns the detailed error message.
        """
        return self.error_message  # Return the detailed error message


# Generalized function for error logging and raising custom exceptions
def log_and_raise_exception(error_message, error_detail: sys):
    """
    This function logs the error message using logging and raises a CustomException.
    """
    logging.error(error_message)  # Log the error message using the imported logger
    raise CustomException(error_message, error_detail)  # Raise the custom exception with detailed error


class ModelTrainer:
    def __init__(self):
        logging.debug("ModelTrainer initialized.")
        self.model_trainer_config = {"trained_model_file_path": os.path.join("artifacts", "model.pkl")}
        self.models = {
            "RandomForest": RandomForestRegressor(n_estimators=100),
            "DecisionTree": DecisionTreeRegressor(),
            "GradientBoosting": GradientBoostingRegressor(),
            "LinearRegression": LinearRegression(),
            "KNeighbors": KNeighborsRegressor(),
            "CatBoost": CatBoostRegressor(iterations=100, learning_rate=0.1, depth=6),
            "AdaBoost": AdaBoostRegressor(),
            "XGBoost": XGBRegressor(n_estimators=100)
        }

    def save_model(self, file_path, model):
        try:
            logging.debug(f"Attempting to save model at {file_path}")
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'wb') as f:
                pickle.dump(model, f)
            logging.debug(f"Model saved at {file_path}")
        except Exception as e:
            logging.error(f"Error saving model: {str(e)}")
            raise CustomException(f"Error saving model: {str(e)}", sys)

    def evaluate_model(self, model, X_test, y_test):
        """
        Evaluates the model on the test data and returns the R2 score.
        """
        try:
            predicted = model.predict(X_test)
            r2_square = r2_score(y_test, predicted)
            logging.info(f"R2 Score: {r2_square}")
            return r2_square
        except Exception as e:
            logging.error(f"Error during model evaluation: {str(e)}")
            raise CustomException(f"Error during model evaluation: {str(e)}", sys)

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.debug("Splitting the training and test data.")

            X_train, y_train, X_test, y_test = (
                train_array[:, :-1], 
                train_array[:, -1], 
                test_array[:, :-1], 
                test_array[:, -1]
            )

            best_r2_score = float('-inf')
            best_model = None

            logging.debug("Starting model training and evaluation.")
            for model_name, model in self.models.items():
                logging.debug(f"Training {model_name} model.")
                model.fit(X_train, y_train)
                
                # Model evaluation
                r2_square = self.evaluate_model(model, X_test, y_test)
                
                # Track the best performing model
                if r2_square > best_r2_score:
                    best_r2_score = r2_square
                    best_model = model
                    logging.info(f"{model_name} model achieved the best R2 score: {r2_square}")

            if best_model is not None:
                logging.debug(f"Best model is {best_model}. Saving the model.")
                self.save_model(self.model_trainer_config["trained_model_file_path"], best_model)
            else:
                logging.error("No model found with a valid R2 score.")
                raise CustomException("No best model found", sys)

            return best_r2_score
        except Exception as e:
            logging.error(f"Error during model training: {str(e)}")
            raise CustomException(str(e), sys)


# Main execution to test the trainer with multiple models
if __name__ == "__main__":
    # Create a simple synthetic dataset
    X_train = np.random.rand(100, 5)
    y_train = np.random.rand(100)
    X_test = np.random.rand(20, 5)
    y_test = np.random.rand(20)

    # Create ModelTrainer object and train
    model_trainer = ModelTrainer()
    best_r2_score = model_trainer.initiate_model_trainer(np.column_stack([X_train, y_train]), np.column_stack([X_test, y_test]))

    print(f"Best Test R2 Score: {best_r2_score}")
