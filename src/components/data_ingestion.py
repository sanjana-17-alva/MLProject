import os
import sys
from src.exceptions import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass  # Used only when class has only class variables (it's called a decorator)
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', "train.csv")  # Path for the training data
    test_data_path: str = os.path.join('artifacts', "test.csv")    # Path for the testing data
    raw_data_path: str = os.path.join('artifacts', "data.csv")     # Path for the raw data

class DataIngestion:
    def __init__(self): 
        self.ingestion_config = DataIngestionConfig()  # Instantiate the DataIngestionConfig class

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method")  # Logging the start of the method
        try:
            # Read the dataset from a CSV file (you can change this to read from MongoDB or any other source)
            df = pd.read_csv('Notebook/Data/StudentsPerformance.csv')  

            logging.info('Read the dataset as dataframe')  

            # Ensure the artifacts directory exists
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)  

            # Save the raw data to raw_data_path
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Train-test split initiated")  # Log the start of train-test split

            # Split the data into train and test sets
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)  # 80% training, 20% testing

            # Save the train and test sets
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)  
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)  

            logging.info("Ingestion of the data is completed")  # Log the completion of data ingestion

            # Return the paths to the train and test data
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )

        except Exception as e:
            # Raise a custom exception with the error message
            raise CustomException("Error in data ingestion: " + str(e))  


#to check if your code is running
# if __name__== "__main__":
#     obj = DataIngestion()
#     obj.initiate_data_ingestion()

