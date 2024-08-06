import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_file_path: str = os.path.join('artifacts', "preprocessor.pkl")

class DataTransformation:

    def __init__(self):
        self.config = DataTransformationConfig()

    
    def create_preprocessor(self, dataframe: pd.DataFrame):
        logging.info("Preprocessor creation initiated")

        try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            numeric_pipeline = Pipeline(
                steps = [
                    ("imputer", SimpleImputer(strategy="mean")),
                    ("scaler", StandardScaler())
                ]
            )

            categorical_pipeline = Pipeline(
                steps = [
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            logging.info("Numeric pipeline created")
            logging.info("Categorical pipeline created")

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", numeric_pipeline, numerical_columns),
                    ("cat_pipeline", categorical_pipeline, categorical_columns)
                ]
            )

            return preprocessor
         
        except Exception as e:
            logging.error("Error in creating preprocessor")
            raise CustomException(e, sys)
        

    def perform_data_transformation(self, train_path, test_path):
        try:
            # Load train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Train and test data loaded successfully")

            target_column = "maths_score"

            logging.info("Creating preprocessor")
            preprocessor = self.create_preprocessor(train_df)

            input_features_train = train_df.drop(columns=[target_column], axis=1)
            target_train = train_df[target_column]

            input_features_test = test_df.drop(columns=[target_column], axis=1)
            target_test = test_df[target_column]

            # Applying preprocessor on training and testing data
            logging.info("Applying preprocessor on training and testing data")
            input_features_train_transformed = preprocessor.fit_transform(input_features_train)
            input_features_test_transformed = preprocessor.transform(input_features_test)

            train_array = np.c_[input_features_train_transformed, np.array(target_train)]
            test_array = np.c_[input_features_test_transformed, np.array(target_test)]

            save_object(
                file_path=self.config.preprocessor_file_path,
                obj=preprocessor
            )

            logging.info("Saved preprocessor")

            return (
                train_array,
                test_array,
                self.config.preprocessor_file_path
            )
        

        except Exception as e:
            logging.error("Error in data transformation process")
            raise CustomException(e, sys)