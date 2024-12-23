import os
import sys
import dill
import logging
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from src.exceptions import CustomException

# Set up basic logging
logging.basicConfig(level=logging.INFO)

def save_object(file_path, obj):
    """
    Saves the object to the specified file path using dill.
    Creates the directories if they do not exist.

    Args:
    - file_path (str): The path to save the object.
    - obj (object): The object to save.
    """
    try:
        if obj is None:
            raise ValueError("Cannot save None object.")
        
        # Ensure the directory exists
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        # Save the object using dill
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)

        logging.info(f"Object saved successfully at {file_path}")

    except Exception as e:
        logging.error(f"Error saving object at {file_path}: {e}")
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train, X_test, y_test, models, params, cv=3, n_jobs=3, verbose=1, refit=False):
    """
    Evaluates the provided models using GridSearchCV for hyperparameter tuning,
    and returns a report with the R2 score for each model.

    Args:
    - X_train (ndarray): Training features.
    - y_train (ndarray): Training labels.
    - X_test (ndarray): Testing features.
    - y_test (ndarray): Testing labels.
    - models (dict): Dictionary of models to evaluate.
    - params (dict): Dictionary of parameter grids for each model.
    - cv (int): Number of cross-validation splits (default: 3).
    - n_jobs (int): Number of jobs to run in parallel (default: 3).
    - verbose (int): Verbosity level (default: 1).
    - refit (bool): Whether to refit on the best parameters (default: False).

    Returns:
    - dict: A dictionary with model names as keys and their corresponding R2 scores as values.
    """
    try:
        report = {}

        # Iterate through models and evaluate
        for model_name, model in models.items():
            param_grid = params.get(model_name, {})

            logging.info(f"Training model: {model_name}")
            gs = GridSearchCV(model, param_grid, cv=cv, n_jobs=n_jobs, verbose=verbose, refit=refit)
            gs.fit(X_train, y_train)

            # Set the best parameters and fit the model
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            # Predict and calculate scores
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            report[model_name] = test_model_score

            logging.info(f"Model: {model_name} - Test Score: {test_model_score}")

        return report

    except Exception as e:
        logging.error(f"Error in model evaluation: {e}")
        raise CustomException(e, sys)

def load_object(file_path):
    """
    Loads a saved object from the specified file path using dill.

    Args:
    - file_path (str): The path to load the object from.

    Returns:
    - object: The loaded object.
    """
    try:
        with open(file_path, 'rb') as file_obj:
            return dill.load(file_obj)

    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        raise CustomException(f"File not found: {file_path}", sys)

    except Exception as e:
        logging.error(f"Error loading object from {file_path}: {e}")
        raise CustomException(e, sys)
