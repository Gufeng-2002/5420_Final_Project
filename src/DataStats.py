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