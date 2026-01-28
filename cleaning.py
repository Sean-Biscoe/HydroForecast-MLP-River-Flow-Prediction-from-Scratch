import numpy as np
import pandas as pd


#Function to remove the outliers
def remove_outliers(values, mean_value, standard_deviation):

    # Create a copy of input values to prevent modification of original data
    cleaned_values = values.copy()

    # Convert to numpy array if not already
    cleaned_values = np.array(cleaned_values, dtype=float)
    
    # Relace -999 or 'a' with NaN
    cleaned_values[(cleaned_values == -999) | (cleaned_values == "a")] = np.nan
    
    # Replace values beyond 4 standard deviations from the mean with NaN
    cleaned_values[(cleaned_values < (mean_value - 3 * standard_deviation)) | 
                   (cleaned_values > (mean_value + 3 * standard_deviation))] = np.nan
    
    # Convert to pandas Series for further analysis
    cleaned_values = pd.Series(cleaned_values)

    # Calculate Interquartile Range (IQR) for further outlier detection
    Q1 = cleaned_values.quantile(0.25)
    Q3 = cleaned_values.quantile(0.75)
    IQR = Q3 - Q1

    # Define the lower and upper bounds for outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Filter out values outside IQR bounds and replace with NaN
    cleaned_values[(cleaned_values <= lower_bound) | (cleaned_values >= upper_bound)] = np.nan
    
    return cleaned_values
