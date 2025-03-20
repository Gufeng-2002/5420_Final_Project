# statistics functions
import pandas as pd
from scipy.stats import kurtosis, shapiro, skew

def distribution_stats(data):
    """Calculate summary statistics, skewness, kurtosis, and Shapiro-Wilk normality test for numerical variables.
    
    Args:
        data (pd.DataFrame): A pandas dataframe with numerical variables.
    
    Returns:
        pd.DataFrame: Summary statistics including mean, std, skewness, kurtosis, and Shapiro-Wilk test results.
    """
    
    # Standard descriptive statistics
    common_stats = data.describe()
    
    # Compute Skewness
    skewness = data.select_dtypes(include="number").skew()
    skewness.name = "skewness"
    
    # Compute Kurtosis
    kurt = data.select_dtypes(include="number").apply(lambda x: kurtosis(x, nan_policy='omit', fisher=True))  # Fisher=True gives excess kurtosis
    kurt.name = "kurtosis"
    
    # Compute Shapiro-Wilk Test (Test for normality)
    shapiro_results = data.select_dtypes(include="number").apply(lambda x: shapiro(x.dropna())[1])  # Extract p-value
    shapiro_results.name = "shapiro_p_value"
    
    # Convert series to DataFrame and format
    skewness = pd.DataFrame(skewness).T
    kurt = pd.DataFrame(kurt).T
    shapiro_results = pd.DataFrame(shapiro_results).T
    
    # Concatenate all results together
    result = pd.concat([common_stats, skewness, kurt, shapiro_results], axis=0)
    
    # check the skewness of the numerical variables
    if abs(result.loc["skewness", :]).max() > 1:
        print("There is at least one variable that is highly skewed")
    else:
        print("There is no variable that is highly skewed")
        
    # check the kurtosis of the numerical variables
    if result.loc["kurtosis", :].max() > 3:
        print("There is at least one variable that has more outliers")
    elif result.loc["kurtosis", :].max() < -3:
        print("There is at least one variable that has fewer outliers")
    else:
        print("There is no variable that has more or fewer outliers")
        
    # check the normality of the numerical variables
    if result.loc["shapiro_p_value", :].max() < 0.05:
        print("There is no vairable that is normally distributed")
    else:
        print("There is at least one variable that is normally distributed")
    
    return result

def numerical_correlation(data):
    """Calculate the correlation matrix for numerical variables.
    
    Args:
        data (pd.DataFrame): A pandas dataframe with numerical variables.
    
    Returns:
        pd.DataFrame: A correlation matrix.
    """
    
    # check the correlation coefficients among the numerical variables
    raw_corr_matrix = data.select_dtypes(include="number").corr()

    # Define a function to style cells
    def highlight_threshold(val, threshold = None):
        """
        Highlight values above a threshold in red.
        """
        if abs(val) > 0.9:
            color = 'background-color: red' 
        elif abs(val) > 0.8:
            color = 'background-color: orange'
        elif abs(val) > 0.7:
            color = 'background-color: yellow'
        elif abs(val) > 0.5:
            color = 'background-color: green' 
        else:
            color = ""
        return color
        
    # Apply the style to the correlation matrix and show
    return raw_corr_matrix.style.applymap(
        lambda val: highlight_threshold(val)
    )

# split the orginal data into train and test data sets
def split_train_test(raw_data, frac = 0.8, store_path = "../data"):
    """Split the raw data into train and test sets randomly and store them 
    in the data directory.

    Args:
        rawdata (_type_): a pandas data frame
    """
    raw_data_train = raw_data.sample(frac=frac, random_state=42)
    raw_data_test = raw_data.drop(raw_data_train.index)
    
    store_path = store_path
    raw_data_train.to_csv(store_path + f"/raw_data_train.csv")
    raw_data_test.to_csv(store_path + f"/raw_data_test.csv")
    
    print(f"The test and train data sets are stored into path: {store_path}")
    # check the results
    print(f"The train data set has {raw_data_train.shape[0]} rows and {raw_data_train.shape[1]} columns")
    print(f"The test data set has {raw_data_test.shape[0]} rows and {raw_data_test.shape[1]} columns")
    print(f"The original data set has {raw_data.shape[0]} rows and {raw_data.shape[1]} columns")
    
    return None

def cols_categorize(data):
    """Fetch the column names for different data types: numerical and categorical

    Args:
        dataframe (_type_): a data frame
        
    Returns:       
        numerical_cols (_type_): a list of numerical column names
        categorical_cols (_type_): a list of categorical column names
    """
    numerical_cols = data.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = data.select_dtypes(include=["object"]).columns.tolist()
    
    categorical_cols = [col for col in categorical_cols if data[col].nunique() < 50]  # Adjust threshold
    
    print("Numerical Columns:\n", '; '.join(numerical_cols))
    print("Categorical Columns:\n", '; '.join(categorical_cols))
    
    return numerical_cols, categorical_cols

# the mapping relations for the three categorical variables that need simple encoding
income_mapping = {
    'under $11k': 1,
    '$11-$25k': 2,
    "$25-$50k": 3,
    ">$50k": 4
}

dnr_mapping = {
    'no dnr': 1,
    'dnr after sadm': 2,
    'dnr before sadm': 3   
}

sfdm2_mapping = {
    "no(M2 and SIP pres)": 1,
    "adl>=4 (>=5 if sur)": 2,
    "SIP>=30": 3,
    "Coma or Intub": 4,
    "<2 mo. follow-up": 5
}

cat_sim_mappings = {
    'income': income_mapping,
    'dnr': dnr_mapping,
    'sfdm2': sfdm2_mapping
}


