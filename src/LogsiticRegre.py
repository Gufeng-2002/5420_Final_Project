import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import accuracy_score, classification_report
from scipy.special import expit  # More stable sigmoid function


# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Custom design matrix mapping function
def truncated_power(x, knot, degree):
    return np.where(x > knot, (x - knot) ** degree, 0.0)

def transform_column_with_splines(x_col, degree=3, knots=None, include_intercept=False):
    """
    Transform a single column using spline basis expansion.
    """
    base = [x_col**d for d in range(1, degree + 1)]
    if include_intercept:
        base = [np.ones_like(x_col)] + base
    if knots:
        for knot in knots:
            base.append(truncated_power(x_col, knot, degree))
    return np.column_stack(base)

def transform_matrix_with_splines(X, knots_dict = None, degree=3, include_intercept=False):
    """
    Apply spline transformation to all columns in X with different knots per column.

    Args:
        X: (n, p) input matrix
        knots_dict: dict mapping column index to list of knots for that column
        degree: spline degree (default = 3 for cubic)
        include_intercept: whether to include an intercept term per column (default = False)

    Returns:
        Transformed design matrix (n, ?)
    """
    X = np.asarray(X)
    n, p = X.shape
    transformed_columns = []
    
    # Transform X for splines with transform_matrix_with_splines : includes intercept,
    quantile_knots = [0.1, 0.9]    # IMPORTANT: assuming these are corresponding quantile values

    knots_dict = {
       i: quantile_knots for i in range(X.shape[1])
    }

    for j in range(p):
        x_col = X[:, j]
        knots = knots_dict.get(j, [])  # empty if not specified
        X_j_spline = transform_column_with_splines(x_col, degree, knots, include_intercept)
        transformed_columns.append(X_j_spline)

    return np.hstack(transformed_columns)


# Logistic loss function (negative log-likelihood)
def average_likelihood_fn(beta, X, y):
    
    # z for linear prediction of logit transformed probability being 1
    z = X @ beta
    
    # p for predicted probability of y being 1
    p = sigmoid(z) # a function of X and beta
    
    # given data and beta, the likelihood of this beta value is 
    epsilon = 1e-9 # to avoid log(0)
    neg_likelihood_beta = -np.mean(y * np.log(p + epsilon) + (1 - y) * np.log(1 - p + epsilon))
    return neg_likelihood_beta # the beta that minimizes this is the best fit on the given data

# Fit model
def fit_logistic(transformed_X, y):
    
    # transform/map the design matrix to a higher dimentional space for splines
    
    # set an initial value to beta vector (0 for now)
    beta_init = np.zeros(transformed_X.shape[1])
    
    # using optimization function from scipy to minimize the negative log-likelihood
    result = minimize(average_likelihood_fn, beta_init, args=(transformed_X, y), method='BFGS')
    
    # return MLE beta (average likelihood function minimized)
    mle_beta = result.x
    return mle_beta