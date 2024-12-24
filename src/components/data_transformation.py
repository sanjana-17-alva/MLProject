import os
import sys
import numpy as np
from dataclasses import dataclass
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from src.logger import logging
from src.exceptions import CustomException
from src.utils import save_object
from src.components.Data_Ingestion import DataIngestion  # Corrected import

@dataclass
class DataTransformationConfig:
    #saving the preprocessor object into a pickle file 
    #os.path.join() ensures that the file path is correctly formed, especially if you're working with multiple OS platforms.
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()  # Instantiate configuration class

    def get_data_transformer_object(self):
        #actual data trnsformation happens here 
        try:
            numerical_features = ["writing score", "reading score"]
            categorical_features = ['gender', 'race/ethnicity', 
                                    'parental level of education', 
                                    'lunch', 'test preparation course']

            # Numerical pipeline
            numerical_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy="median")),
                    ('scaler', StandardScaler())
                ]
            )
            logging.info('Numerical features are standardized')

            # Categorical pipeline
            categorical_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    # Corrected sparse argument to sparse_output=False
                    ('one_hot_encoder', OneHotEncoder(sparse_output=False)),
                    ('scaler', StandardScaler())
                ]
            )

            logging.info('Categorical columns are encoded')

            # Combine pipelines using ColumnTransformer
            preprocessor = ColumnTransformer(
                transformers=[
                    ('numerical_pipeline', numerical_pipeline, numerical_features),
                    ('categorical_pipeline', categorical_pipeline, categorical_features)
                ]
            )

            return preprocessor
        except Exception as e:
            # Pass the sys argument to CustomException for detailed traceback
            raise CustomException("Error in data transformation: " + str(e), sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            # Read training and testing dataframes
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Reading of train and test datasets is completed")
            logging.info("Obtaining the preprocessing object")

            # Get the preprocessing pipeline object
            preprocessing_obj = self.get_data_transformer_object()

            # Define target column and numerical features
            target_column_name = "math score"

            # Separate input and target features
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]
            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info(f"Applying preprocessing object on training and testing data")

            # Apply the preprocessing object (fit and transform on train data, transform on test data)
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # Combine transformed input features with target features
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Preprocessing object saved successfully.")

            # Save preprocessor object using save_object utility function
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            # Return transformed arrays and preprocessor object file path
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            # Pass sys argument to CustomException for detailed traceback
            raise CustomException(e, sys)


# # to check the code
# if __name__ == '__main__':
#     obj = DataIngestion()  # Correct instantiation

#     # Call the data ingestion method
#     train_data, test_data = obj.initiate_data_ingestion()

#     # Perform data transformation
#     data_transformation = DataTransformation()
#     data_transformation.initiate_data_transformation(train_data, test_data)