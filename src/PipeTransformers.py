# General
import numpy as np
import pandas as pd
import importlib
# Scikit-learn components
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.utils.validation import check_is_fitted
import DataStats



# Custom FunctionTransformer with get_feature_names_out
class CustomFunctionTransformer(FunctionTransformer):
    def __init__(self, func, feature_names=None, **kwargs):
        super().__init__(func=func, **kwargs)
        self.feature_names = feature_names

    def fit(self, X, y=None):
        self.feature_names_in_ = X.columns if isinstance(X, pd.DataFrame) else None
        return super().fit(X, y)

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "feature_names_in_")
        if self.feature_names:
            return self.feature_names
        return self.feature_names_in_
    
    
# prepare a sample data set
data = pd.read_csv('../data/original_data.csv').drop('Unnamed: 0', axis=1)
data = data[:100]
num_cols, cat_cols = DataStats.cols_categorize(data)

# some categorical columns should be encoded by simple encoding, not one-hot encoding
cat_sim_cols = ['income', 'dnr', 'sfdm2'] # need to be encoded by simple encoding
cat_oh_cols = list(set(cat_cols) - set(cat_sim_cols)) # one-hot encoding


# before running the pipeline, we should remove the missing values in the target column
def create_imputer_encoder_pipelines():
    
    """
    Creates a collection of pipelines for imputing and encoding numerical and categorical data.
    This function generates various combinations of imputers and encoders for preprocessing data.
    It includes pipelines for handling missing values in numerical and categorical columns using
    different strategies, as well as encoding categorical variables using one-hot encoding or 
    custom mappings.
    Returns:
        dict: A dictionary where keys are integers representing pipeline IDs, and values are 
              `ColumnTransformer` objects that combine the specified imputers and encoders for 
              numerical and categorical columns.
    Pipelines:
        - Numerical Imputers:
            - Mean imputation
            - Median imputation
            - Constant value imputation (e.g., -1)
            - K-Nearest Neighbors (KNN) imputation
            - Iterative imputation using regression models
        - Categorical Imputers:
            - Most frequent value imputation
            - Constant value imputation (e.g., "Unknown")
        - Encoders:
            - One-hot encoding for categorical variables
            - Custom mapping-based encoding for specific categorical variables
    Notes:
        - The function uses `ColumnTransformer` to combine the pipelines for numerical and 
          categorical columns.
        - The categorical mapping-based encoder uses a custom function transformer to apply 
          predefined mappings to specific columns.
        - The function returns all possible combinations of numerical imputers, categorical 
          imputers, and encoders.
    """
    
    # Test for restrcturing the pipeline: spliting the imputer and encoder parts
    # different numerical imputing methods
    num_imputer_1 = Pipeline([
        ("imputer", SimpleImputer(strategy="mean"))
    ])
    num_imputer_2 = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))  # Uses median instead of mean
    ])

    num_imputer_3 = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value=-1))  # Replace NaN with -1
    ])

    num_imputer_4 = Pipeline([
        ("imputer", KNNImputer(n_neighbors=3))  # Uses 3 nearest neighbors
    ])

    num_imputers = [num_imputer_1, num_imputer_2, num_imputer_3, num_imputer_4]

    # different categorical imputing methods with one-hot encoding
    cat_oh_imputer_1 = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        # ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    cat_oh_imputer_2 = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),  # Replace NaN with "Unknown"
        # ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    cat_oh_imputers = [cat_oh_imputer_1, cat_oh_imputer_2]

    cat_oh_encoder = Pipeline([
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    # categorical imputing methods with simple encoding
    cat_sim_imputer_1 = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
    ])

    cat_sim_imputer_2 = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value= 'unknown')),  # Replace NaN with "Unknown"
    ])

    cat_sim_imputers = [cat_sim_imputer_1, cat_sim_imputer_2]

    # customized encoders for some categorical variables
    cat_sim_maps = DataStats.cat_sim_mappings

    # Define mapping function
    def cat_sim_maps_func(X):
        X = X.copy()
        for col, mapping in cat_sim_maps.items():
            X[col] = X[col].map(mapping)
        return X

    # Add FunctionTransformer to the existed pipeline
    encode_dict_transformer = CustomFunctionTransformer(cat_sim_maps_func) # waiting to be added into the pipeline

    cat_sim_encoder = Pipeline([
        ("cat_sim_encoder", encode_dict_transformer)
    ])


    # combining them to create all possible pipelines (2 * 5 = 10)
    imputer_encoders = [
        ColumnTransformer(
            [("num_imputer", num_imputer, num_cols),
            ("cat_on_imputer", cat_oh_imputer, cat_oh_cols),
            ("cat_sim_imputer", cat_sim_imputer, cat_sim_cols),
            ("cat_oh_encoder", cat_oh_encoder, cat_oh_cols),
            ("cat_sim_encoder", cat_sim_encoder, cat_sim_cols)]
            ) 
    for cat_oh_imputer in cat_oh_imputers
    for num_imputer in num_imputers
    for cat_sim_imputer in cat_sim_imputers]

    imputer_encoders = {i: imputer for i, imputer in enumerate(imputer_encoders)}
        
    return imputer_encoders

# After creating imputer_encoders, use the "inputer_transform" function to transform the data with 
# a specific imputer_encoder.

# a function envokes the imputer to transform the data and convert it to a data frame
def imputer_encoder_transform(imputer_encoder, data):
    
    """Transforms the input data using the provided imputer encoder, handling missing values 
    by filling them with the rounded mean of the respective columns.
    Parameters:
    -----------
    imputer_encoder : sklearn.impute.BaseImputer
        An instance of a scikit-learn imputer or encoder that supports the `fit`, `transform`, 
        and `get_feature_names_out` methods.
    data : pandas.DataFrame or numpy.ndarray
        The input data to be transformed. It should be compatible with the imputer_encoder.
    Returns:
    --------
    pandas.DataFrame
        A DataFrame containing the transformed data with missing values filled using the 
        rounded mean of the respective columns.
    Notes:
    ------
    - The function first fits the imputer_encoder on the input data and then transforms it.
    - Missing values in the transformed data are identified and replaced with the rounded 
      mean of the corresponding column.
    - The transformed data is returned as a pandas DataFrame with column names derived 
      from the imputer_encoder.
    """
    
    # fit the imputer on the data
    imputer_encoder.fit(data)
    # transform the data using the fitted imputer
    transformed_data = imputer_encoder.transform(data)
    # convert the transformed data to a DataFrame
    transformed_data = pd.DataFrame(transformed_data, columns=imputer_encoder.get_feature_names_out())
    
    # fill in the missing values in the transfored data with the rounded up mean of the column
    # check the columns that have missing values in the transformed data
    missing_cols = transformed_data.columns[transformed_data.isnull().any()].tolist()
    missing_cols
    for missing_col in missing_cols:
        mean_value = transformed_data[missing_col].mean().round()
        transformed_data[missing_col] = transformed_data[missing_col].fillna(mean_value)
        
    # return the transformed data
    return transformed_data
