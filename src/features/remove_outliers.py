import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy
from sklearn.neighbors import LocalOutlierFactor  # pip install scikit-learn

# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------
df=pd.read_pickle("../../data/interim/01_data_processed.pkl")
outlier_column=list(df.columns[:6])
df.info()

# --------------------------------------------------------------
# Plotting outliers
# --------------------------------------------------------------
plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20,5)
plt.rcParams["figure.dpi"] =100
df[["gyr_y","label"]].boxplot(by="label", figsize=(20,10)) # box plot is a method
#for graphically depicting groups of numerical data through their quartiles.
df[outlier_column[:3] + ["label"]].boxplot(by="label", figsize=(20,10),layout=(1,3))
df[outlier_column[3:] + ["label"]].boxplot(by="label", figsize=(20,10),layout=(1,3))


def plot_binary_outliers(dataset, col, outlier_col, reset_index):
    """ Plot outliers in case of a binary outlier score. Here, the col specifies the real data
    column and outlier_col the columns with a binary value (outlier or not).

    Args:
        dataset (pd.DataFrame): The dataset
        col (string): Column that you want to plot
        outlier_col (string): Outlier column marked with true/false
        reset_index (bool): whether to reset the index for plotting
    """

    dataset = dataset.dropna(axis=0, subset=[col, outlier_col])# In arguments if axis=0 then drop rows containing NA values and subset is used to include columns with the axis
    #The above statement will drop rows from the dataframe where either
    #the column 'col' or the column 'outlier_col' contains missing value (NaN)
    dataset[outlier_col] = dataset[outlier_col].astype("bool") # It will convert the value of outlier_col column into boolean i.e True or false
    # From the above line the column outlier_col will contain boolean values (True or False) instead of its original data type.
    if reset_index:
        dataset = dataset.reset_index()

    fig, ax = plt.subplots()

    plt.xlabel("samples")
    plt.ylabel("value")

    # Plot non outliers in default color
    ax.plot(
        dataset.index[~dataset[outlier_col]],
        dataset[col][~dataset[outlier_col]],
        "+",
    )
    # Plot data points that are outliers in red
    ax.plot(
        dataset.index[dataset[outlier_col]],
        dataset[col][dataset[outlier_col]],
        "r+",
    )

    plt.legend(
        ["outlier " + col, "no outlier " + col],
        loc="upper center",
        ncol=2,
        fancybox=True,
        shadow=True,
    )
    plt.show()
    
# --------------------------------------------------------------
# Interquartile range (distribution based)
# --------------------------------------------------------------

def mark_outliers_iqr(dataset, col):
    """Function to mark values as outliers using the IQR method.

    Args:
        dataset (pd.DataFrame): The dataset
        col (string): The column you want apply outlier detection to

    Returns:
        pd.DataFrame: The original dataframe with an extra boolean column 
        indicating whether the value is an outlier or not.
    """

    dataset = dataset.copy()

    Q1 = dataset[col].quantile(0.25) #Q1 variable depicts the threshold below which 25% of the data points in the column 'col' fall
    Q3 = dataset[col].quantile(0.75)
    #in simple terms, the quantile() function tells you where 
    # certain percentages of your data fall when they're arranged in order.
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    dataset[col + "_outlier"] = (dataset[col] < lower_bound) | (
        dataset[col] > upper_bound
    )

    return dataset

# Insert IQR function


# Plot a single column
col="acc_x"
dataset = mark_outliers_iqr(df, col)
plot_binary_outliers(dataset=dataset,col=col, outlier_col=col+"_outlier",reset_index=True)

# Loop over all columns
for col in outlier_column:
    dataset = mark_outliers_iqr(df,col)
    plot_binary_outliers(dataset=dataset, col=col, outlier_col=col+"_outlier",reset_index=True)

# --------------------------------------------------------------
# Chauvenets criteron (distribution based)
# --------------------------------------------------------------

# Check for normal distribution
df[outlier_column[:3] + ["label"]].plot.hist(by="label", figsize=(20,20),layout=(3,3))
df[outlier_column[3:] + ["label"]].plot.hist(by="label", figsize=(20,20),layout=(3,3))
#Note:- Chauvenet's criterion is only applicable to datasets that are normally distributed.
# If your dataset is not normally distributed, this method may not be suitable for identifying outliers.

# Insert Chauvenet's function
def mark_outliers_chauvenet(dataset, col, C=2):
    """Finds outliers in the specified column of datatable and adds a binary column with
    the same name extended with '_outlier' that expresses the result per data point.
    
    Taken from: https://github.com/mhoogen/ML4QS/blob/master/Python3Code/Chapter3/OutlierDetection.py

    Args:
        dataset (pd.DataFrame): The dataset
        col (string): The column you want apply outlier detection to
        C (int, optional): Degree of certainty for the identification of outliers given the assumption 
                           of a normal distribution, typicaly between 1 - 10. Defaults to 2.

    Returns:
        pd.DataFrame: The original dataframe with an extra boolean column 
        indicating whether the value is an outlier or not.
    """

    dataset = dataset.copy()
    # Compute the mean and standard deviation.
    mean = dataset[col].mean()
    std = dataset[col].std()
    N = len(dataset.index)
    criterion = 1.0 / (C * N)

    # Consider the deviation for the data points.
    deviation = abs(dataset[col] - mean) / std

    # Express the upper and lower bounds.
    low = -deviation / math.sqrt(C)
    high = deviation / math.sqrt(C)
    prob = []
    mask = []

    # Pass all rows in the dataset.
    for i in range(0, len(dataset.index)):
        # Determine the probability of observing the point
        prob.append(
            1.0 - 0.5 * (scipy.special.erf(high.iloc[i]) - scipy.special.erf(low.iloc[i]))
        )
        # And mark as an outlier when the probability is below our criterion.
        mask.append(prob[i] < criterion)
    dataset[col + "_outlier"] = mask
    return dataset

# Loop over all columns
for col in outlier_column:
    dataset = mark_outliers_chauvenet(df,col)
    plot_binary_outliers(dataset=dataset,col=col,outlier_col=col+"_outlier",reset_index=True)
    

# --------------------------------------------------------------
# Local outlier factor (distance based)
# --------------------------------------------------------------

# Insert LOF function
def mark_outliers_lof(dataset, columns, n=20):
    """Mark values as outliers using LOF

    Args:
        dataset (pd.DataFrame): The dataset
        col (string): The column you want apply outlier detection to
        n (int, optional): n_neighbors. Defaults to 20.
    
    Returns:
        pd.DataFrame: The original dataframe with an extra boolean column
        indicating whether the value is an outlier or not.
    """
    
    dataset = dataset.copy()

    lof = LocalOutlierFactor(n_neighbors=n)
    data = dataset[columns]
    outliers = lof.fit_predict(data)
    X_scores = lof.negative_outlier_factor_

    dataset["outlier_lof"] = outliers == -1
    return dataset, outliers, X_scores

# Loop over all columns
dataset, outliers, x_scores = mark_outliers_lof(df,outlier_column)
for col in outlier_column:
    plot_binary_outliers(dataset=dataset, col=col, outlier_col="outlier_lof",reset_index=True)

# --------------------------------------------------------------
# Check outliers grouped by label
# --------------------------------------------------------------
label="row"
for col in outlier_column:
    dataset=mark_outliers_iqr(df[df["label"]==label], col)
    plot_binary_outliers(dataset,col,col+"_outlier",reset_index=True)
for col in outlier_column:
    dataset=mark_outliers_chauvenet(df[df["label"]==label], col)
    plot_binary_outliers(dataset,col,col+"_outlier",reset_index=True)
    
dataset, outliers, x_scores = mark_outliers_lof(df[df["label"]==label],outlier_column)
for col in outlier_column:
    plot_binary_outliers(dataset=dataset, col=col, outlier_col="outlier_lof",reset_index=True)

# --------------------------------------------------------------
# Choose method and deal with outliers
# --------------------------------------------------------------

# Test on single column
col="gyr_z"
dataset=mark_outliers_chauvenet(df,col=col)
dataset[dataset["gyr_z_outlier"]]
dataset.loc[dataset["gyr_z_outlier"], "gyr_z"]=np.nan # it will set the values in the column "gyr_z" to NaN (missing values)
#for rows where the corresponding value in the column "gyr_z_outlier" is True.

# Create a loop
outliers_removed_df=df.copy()
for col in outlier_column:
    for label in df["label"].unique():
        dataset=mark_outliers_chauvenet(df[df["label"]==label],col)
        
        # Replace values marked as outliers with NaN    
        dataset.loc[dataset[col+"_outlier"],col]=np.nan
        #this line of code replaces values in the column col of 
        # the DataFrame dataset with NaN for rows where the condition dataset[col+"_outlier"] is True
       
        # Update the column in the original dataframe
        outliers_removed_df.loc[(outliers_removed_df["label"]==label),col]=dataset[col]
        # replaces values in the column col of the DataFrame outliers_removed_df with values from
        # the column col of the DataFrame dataset for rows where the value in the "label" column of outliers_removed_df equals the specified label
        n_outliers=len(dataset)-len(dataset[col].dropna())
        print(f"Removed {n_outliers} from {col} for {label}")
                
outliers_removed_df.info()        
# --------------------------------------------------------------
# Export new dataframe
# --------------------------------------------------------------

outliers_removed_df.to_pickle("../../data/interim/02_outliers_removed_chauvenets.pkl")
#in this file we have removed outliers and the outliers values are filled with NaN so there are gaps in dataframe
# so most algorithm doesn't function properly with the NaN data , this will be recorrected by the help of feature engineering