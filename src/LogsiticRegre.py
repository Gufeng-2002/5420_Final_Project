import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import accuracy_score, classification_report
from scipy.special import expit  # More stable sigmoid function


# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

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
def fit_logistic(transformed_X, y, beta_init = None):
    
    # transform/map the design matrix to a higher dimentional space for splines
    
    # set an initial value to beta vector (0 for now)
    beta_init = np.zeros(transformed_X.shape[1])
    
    # using optimization function from scipy to minimize the negative log-likelihood
    result = minimize(average_likelihood_fn, beta_init, args=(transformed_X, y), method='BFGS')
    
    # return MLE beta (average likelihood function minimized)
    mle_beta = result.x
    return mle_beta

# transform function - "transform_matrix_with_splines"
# Custom design matrix mapping function
def truncated_power(x, knot, degree):
    return np.where(x > knot, (x - knot) ** degree, 0.0)

def transform_column_with_splines(x_col, degree=2, knots=None, include_intercept=False):
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

def transform_matrix_with_splines(X, knots_dict = None, degree=2, include_intercept=False):
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
    quantile_knots = [0.9]    # IMPORTANT: assuming these are corresponding quantile values

    knots_dict = {
       i: quantile_knots for i in range(X.shape[1])
    }

    for j in range(p):
        x_col = X[:, j]
        knots = knots_dict.get(j, [])  # empty if not specified
        X_j_spline = transform_column_with_splines(x_col, degree, knots, include_intercept)
        transformed_columns.append(X_j_spline)

    return np.hstack(transformed_columns)

class Samples:
    # bootstraped samples from X_sim and y_sim
    def bootstrap_samples(self, X, y, B = 1000):
        """
        Generate bootstrap samples from the given data.

        Args:
            X (numpy.ndarray): The input feature matrix of shape (n_samples, n_features).
            y (numpy.ndarray): The target vector of shape (n_samples,).
            B (int): The number of bootstrap samples to generate (default is 1000).

        Returns:
            list: A list of tuples, where each tuple contains a bootstrap sample of X and the corresponding y.
        """
        n, p = X.shape
        # Create an array to hold the bootstrap samples
        bootstrap_samples = []
        for _ in range(B):
            idx = np.random.choice(n, size=n, replace=True)
            Xb, yb = X[idx], y[idx]
            bootstrap_samples.append((Xb, yb))
            
        return bootstrap_samples
    
    # generate bootstrap estimates of beta
    def bootstrap_estimates(self, X, y, transformed_degree = 2, B=1000):
        """
        Generate bootstrap estimates of beta.

        Args:
            X (numpy.ndarray): The input feature matrix of shape (n_samples, n_features).
            y (numpy.ndarray): The target vector of shape (n_samples,).
            B (int): The number of bootstrap samples to generate (default is 1000).

        Returns:
            numpy.ndarray: An array of shape (B, n_features) containing the bootstrap estimates of beta.
        """
        bootstrap_B_samples = self.bootstrap_samples(X, y, B)
        bootstrap_estimates = np.zeros((B, X.shape[1]))
        
        for i in range(B):
            Xb, yb = bootstrap_B_samples[i]
            # transformed_Xb = transform_matrix_with_splines(Xb, degree = transformed_degree, include_intercept=True)
            beta_b = fit_logistic(Xb, yb)
            bootstrap_estimates[i] = beta_b
            
        return bootstrap_estimates
    

class Posterior:
    def posterior_estimates_density(self, X, y, B = 1000):
        """
        Calculate posterior estimates using likelihood function and bootstrap estimates.
        """
        # mle beta from the original data
        mle_beta = fit_logistic(X, y)
        mle_beta
        
        # bootstrap estimates, prior of beta
        bootstrap_estimator = Samples()
        priors_beta = bootstrap_estimator.bootstrap_estimates(X, y, B=B)

        # Assuming priors_beta and likelihood_boot_betas are defined
        likelihood_boot_betas = []
        
        for i in range(priors_beta.shape[0]):
            beta = priors_beta[i, :]
            likelihood = -average_likelihood_fn(beta, X, y) # caution: the original function is a loss function
            likelihood_boot_betas.append(likelihood)
        
        # Calculate the posterior proportional function values
        posterior_density = np.array(likelihood_boot_betas) * (1 / priors_beta.shape[0])
        
        return priors_beta, posterior_density

    # take the most probable beta estimates
    def posterior_quantile_betas(self, posterior_density_results, beta_estimates, top_quantile = 0.9):
        sorted_indices = np.argsort(posterior_density_results)[::-1]
        top_quantile_percent_indices = sorted_indices[:int(top_quantile * len(sorted_indices))]
        # take the corresponding beta values
        top_quantile_betas = beta_estimates[top_quantile_percent_indices]
        # use the mean of the top 30 percent beta values as the final beta
        final_beta = np.mean(top_quantile_betas, axis=0)
        return final_beta
    
def bayesian_logistic_regression(X, y, B=1000):
    """
    Perform Bayesian logistic regression using bootstrap estimates and posterior density.
    
    Args:
        X (numpy.ndarray): Feature matrix of shape (n_samples, n_features).
        y (numpy.ndarray): Target vector of shape (n_samples,).
        B (int): Number of bootstrap samples (default is 1000).
    
    Returns:
        tuple: Tuple containing:
            - beta_estimates (numpy.ndarray): Estimated coefficients.
            - posterior_density_results (numpy.ndarray): Posterior density results.
    """
    
    # prior of beta
    bootstrap_estimator = Samples()
    bootstrap_estimates_results = bootstrap_estimator.bootstrap_estimates(X, y, B=y.shape[0])

    # posterior of beta
    post_mc = Posterior()
    beta_estimates, posterior_density_results = post_mc.posterior_estimates_density(X, y, B=y.shape[0])

    # expectation of posterior beta
    posterior_beta_exp = post_mc.posterior_quantile_betas(posterior_density_results, beta_estimates, top_quantile=0.9)
    posterior_beta_exp
    
    return posterior_beta_exp

# define a function to predict the target
def predict_p_values(X, beta, prob_threshold = 0.5, return_probs = True):
    """
    Predict the target probabilities using the logistic regression coefficients.
    
    Args:
        X (numpy.ndarray): Feature matrix of shape (n_samples, n_features).
        beta (numpy.ndarray): Coefficients of shape (n_features,).
    
    Returns:
        numpy.ndarray: Predicted probabilities of shape (n_samples,).
    """
    
    # transform the feature matrix
    # transformed_X = transform_matrix_with_splines(X, degree=2, include_intercept=True)
    transformed_X = X
    
    # calculate the logits
    logits = transformed_X @ beta
    
    # calculate the probabilities
    probs = expit(logits)
    
    if not return_probs:
        probs = np.where(probs > prob_threshold, 1, 0)
    
    return probs

def prepare_data(data, target_value = 1):
    # For multi-class classification, we need to convert the target variable into binary
    # Defaultly, only consider the target having value 1
    copy_data = data.copy()
    copy_data['target'] = copy_data['target'].apply(lambda x: 1 if x == target_value else 0)
    
    # take out the feature matrix and target vector
    X = copy_data.drop(columns=['target']).values
    y = copy_data['target'].values
    return X, y

# waiting for the one-vs-rest logistic regression function to be implemented
# fit one-vs-rest logistic regression for multiclass classification
def fit_multiclass_logistic_ovr(data, num_classes):
    models = {}
    mle_models = {}
    bayesian_models = {}
    for target in range(num_classes):
        target = target + 1
        X, y = prepare_data(data, target_value=target)
        mle_beta = fit_logistic(X, y)   # Use your earlier function
        bayesian_beta = bayesian_logistic_regression(X, y, B = y.shape[0])
        # store estimate of beta
        mle_models[target] = mle_beta
        bayesian_models[target] = bayesian_beta
    models["MLE"] = mle_models
    models["Bayesian"] = bayesian_models
    return models

# use the models to predict the test data
def predict_multiclass(data, models, return_probs=True):
    y_pred = {}
    for target in range(1, 6):
        X, y = prepare_data(data, target_value=target)
        # use the MLE model to predict
        mle_beta = models["MLE"][target]
        bayesian_beta = models["Bayesian"][target]
        y_pred[target] = {
            "MLE": predict_p_values(X, mle_beta, prob_threshold = 0.5, return_probs = return_probs),
            "Bayesian": predict_p_values(X, bayesian_beta, prob_threshold = 0.5, return_probs=return_probs)
        }
    return y_pred