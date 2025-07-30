import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from DataTransformation import LowPassFilter
from scipy.signal import argrelextrema
from sklearn.metrics import mean_absolute_error

pd.options.mode.chained_assignment = None


# Plot settings
# plt.style.use("fivethirtyeight")
# plt.rcParams["figure.figsize"] = (20, 5)
# plt.rcParams["figure.dpi"] = 100
# plt.rcParams["lines.linewidth"] = 2


# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------
df=pd.read_pickle("../../data/interim/01_data_processed.pkl")
df=df[df["label"]!="rest"]

acc_r=df["acc_x"] ** 2 + df["acc_y"] ** 2 +df["acc_z"] ** 2 
gyr_r=df["gyr_x"] ** 2 + df["gyr_y"] ** 2 +df["gyr_z"] ** 2 
df["acc_r"]=np.sqrt(acc_r)
df["gyr_r"]=np.sqrt(gyr_r)
# --------------------------------------------------------------
# Split data
# --------------------------------------------------------------
bench_df=df[df["label"]=="bench"]
squat_df=df[df["label"]=="squat"]
row_df=df[df["label"]=="row"]
ohp_df=df[df["label"]=="ohp"]
dead_df=df[df["label"]=="dead"]
# --------------------------------------------------------------
# Visualize data to identify patterns
# --------------------------------------------------------------

# there are heavy and medium sets in the category of dataframe in heavy set there are mostly 5 reps and in medium there are 10

plot_df=squat_df
plot_df[plot_df["set"]==plot_df["set"].unique()[0]]["acc_x"].plot()
plot_df[plot_df["set"]==plot_df["set"].unique()[0]]["acc_y"].plot()
plot_df[plot_df["set"]==plot_df["set"].unique()[0]]["acc_z"].plot()
plot_df[plot_df["set"]==plot_df["set"].unique()[0]]["acc_r"].plot()

plot_df[plot_df["set"]==plot_df["set"].unique()[0]]["gyr_x"].plot()
plot_df[plot_df["set"]==plot_df["set"].unique()[0]]["gyr_y"].plot()
plot_df[plot_df["set"]==plot_df["set"].unique()[0]]["gyr_z"].plot()
plot_df[plot_df["set"]==plot_df["set"].unique()[0]]["gyr_r"].plot()

# --------------------------------------------------------------
# Configure LowPassFilter
# --------------------------------------------------------------

fs=1000/200
Lowpass=LowPassFilter()

# --------------------------------------------------------------
# Apply and tweak LowPassFilter
# --------------------------------------------------------------
bench_set=bench_df[bench_df["set"]==bench_df["set"].unique()[0]]
squat_set=squat_df[squat_df["set"]==squat_df["set"].unique()[0]]
row_set=row_df[row_df["set"]==row_df["set"].unique()[0]]
ohp_set=ohp_df[ohp_df["set"]==ohp_df["set"].unique()[0]]
dead_set=dead_df[dead_df["set"]==dead_df["set"].unique()[0]]

bench_set["acc_y"].plot()
squat_set["acc_y"].plot()
column="acc_y"
Lowpass.low_pass_filter(bench_set,col=column,sampling_frequency=fs,cutoff_frequency=0.4,order=10)[column + "_lowpass"].plot()
# --------------------------------------------------------------
# Create function to count repetitions
# --------------------------------------------------------------
col="acc_r"
data = Lowpass.low_pass_filter(bench_set,col="acc_r",sampling_frequency=fs,cutoff_frequency=0.4,order=10)
indexes=argrelextrema(data[col+"_lowpass"].values,np.greater) #argrelextrema is used to find the indices of local extrema(such as peaks)
peaks=data.iloc[indexes] #The indexes array contains the positions of the peaks identified earlier. By passing indexes to iloc, you're
                        #selecting the rows from data corresponding to these peak positions.


def count_rep( dataset,cutoff=0.4,order=10,column="acc_r"):
   data = Lowpass.low_pass_filter(dataset,col=column,sampling_frequency=fs,cutoff_frequency=cutoff,order=order)
   indexes=argrelextrema(data[column+"_lowpass"].values,np.greater)
   peaks=data.iloc[indexes]
   
#    fig,ax=plt.subplots()
#    plt.plot(dataset[f"{column}_lowpass"])
#    plt.plot(peaks[f"{column}_lowpass"],"o",color="red")
#    ax.set_ylabel(f"{column}_lowpass")
#    exercise=dataset["label"].iloc[0].title()
#    category=dataset["category"].iloc[0].title()
#    plt.title(f"{category} {exercise} : {len(peaks)} Reps")
#    plt.show()
   
   return len(peaks)

count_rep(bench_set,cutoff=0.4)
count_rep(squat_set,cutoff=0.35)
count_rep(row_set,cutoff=0.65,column="gyr_x")
count_rep(ohp_set,cutoff=0.35)
count_rep(dead_set,cutoff=0.4)
# --------------------------------------------------------------
# Create benchmark dataframe
# --------------------------------------------------------------

df["reps"]=df["category"].apply(lambda x:5 if x=="heavy" else 10)
rep_df=df.groupby(["label","category","set"])["reps"].max().reset_index()
rep_df["reps_pred"]=0

for s in df["set"].unique():
    subset=df[df["set"]==s]
    
    column="acc_r"
    cutoff=0.4
    
    if subset["label"].iloc[0]=="squat":
        cutoff=0.35
    if subset["label"].iloc[0]=="row":
        cutoff=0.65
        col="gyr_x"
    if subset["label"].iloc[0]=="ohp":
        cutoff=0.35
        
    reps=count_rep(subset,cutoff=cutoff,column=column)
    
    rep_df.loc[rep_df["set"]==s, "reps_pred"]=reps #it updates the "reps_pred" column of the DataFrame rep_df 
    #where the "set" column matches the current value of s with the calculated reps.
    
                
# --------------------------------------------------------------
# Evaluate the results
# --------------------------------------------------------------
error = mean_absolute_error(rep_df["reps"], rep_df["reps_pred"]).round(2)

plt.figure(figsize=(5, 5))  # Adjust the width and height as needed
rep_df.groupby(["label", "category"])[["reps", "reps_pred"]].mean().plot.bar()
plt.title("Mean Reps vs Predicted Reps")
plt.xlabel("Label and Category")
plt.ylabel("Mean Reps")
plt.tight_layout() 
plt.show()