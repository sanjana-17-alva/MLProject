import os
import sys
from src.exceptions import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass


@dataclass #used only when class has only class variables-its called a decorator
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts',"train.csv")  # Path for the training data
    test_data_path: str = os.path.join('artifacts',"test.csv")    # Path for the testing data
    raw_data_path: str = os.path.join('artifacts',"data.csv")     # Path for the raw data


class DataIngestion:
    def __init__(self): 
        self.ingension_config = DataIngestionConfig()  # Instantiating the DataIngestionConfig class

    def initiate_data_ingestion(self):
        logging.info("Entered the data method or component")  # Logging the start of the method
        try:
            #reading the dataset from anywhere- clipboard,api,database
            #put your MongoDB client here or any DB client can be read from here
            df = pd.read_csv('Notebook/Data/StudentsPerformance.csv')  

            logging.info('Read the dataset as dataframe')  

            #making the artifact folder
            os.makedirs(os.path.dirname(self.ingension_config.train_data_path), exist_ok=True)  # Ensures the directory exists

            # Save the raw data to raw_data_path
            df.to_csv(self.ingension_config.raw_data_path, index=False, header=True)

            logging.info("Train test split initiated")  # Log the start of train-test split

            #train test split
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)  # Splitting data into train and test sets

            #saving the train file
            train_set.to_csv(self.ingension_config.train_data_path, index=False, header=True)  # Save the training data

            test_set.to_csv(self.ingension_config.test_data_path, index=False, header=True)  # Save the testing data

            logging.info("Ingestion of the data is completed")  # Log the completion of data ingestion

            # These paths are returned for further processing like data transformation
            return(
                self.ingension_config.train_data_path,
                self.ingension_config.test_data_path,
            )

        except Exception as e:
            raise CustomException("Error in data ingestion: " + str(e))  # Raise custom exception with error message
        


#to check if your code is running
if __name__== "__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion()

