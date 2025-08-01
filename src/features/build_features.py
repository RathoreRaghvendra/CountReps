import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from DataTransformation import LowPassFilter, PrincipalComponentAnalysis
from TemporalAbstraction import NumericalAbstraction
from FrequencyAbstraction import FourierTransformation
from sklearn.cluster import KMeans
# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------
df=pd.read_pickle("../../data/interim/02_outliers_removed_chauvenets.pkl")
predictor_columns=list(df.columns[:6])
df.info() #this dataframe contains missing values in some of the column bcoz of the outlier function 

#plot settings

plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"]=(20,5)
plt.rcParams["figure.dpi"]=100
plt.rcParams["lines.linewidth"]=2

# --------------------------------------------------------------
# Dealing with missing values (imputation)
# --------------------------------------------------------------
for col in predictor_columns:
    df[col]=df[col].interpolate() #It's mostly used to impute missing values in the data frame or series while preprocessing data
 
df.info() #Now the data is interpolated and doesn't contain missing value
# --------------------------------------------------------------
# Calculating set duration
# --------------------------------------------------------------
df[df["set"]==25]["acc_y"].plot()
df[df["set"]==50]["acc_y"].plot()

duration= df[df["set"]==1].index[-1] - df[df["set"]==1].index[0]
duration.seconds

for s in df["set"].unique():
    start=df[df["set"]==s].index[0]
    stop=df[df["set"]==s].index[-1]
    
    duration=stop-start
    df.loc[(df["set"]==s),"duration"]=duration.seconds

duration_df=df.groupby(["category"])["duration"].mean() #In this category will be groupedby with the mean of the duration

duration_df.iloc[0]/5 # /5 bcoz we have seen in graphs that one with heavy set is doing 5 reps only so to calculate the time of single rep this is done
duration_df.iloc[1]/10

# --------------------------------------------------------------
# Butterworth lowpass filter
# --------------------------------------------------------------
df_lowpass=df.copy()
Lowpass=LowPassFilter() #creating object of LowPassFilter class

fs=1000/200 #we have taken 1000ms/200ms bcoz in the data sampling time we have taken rule="200ms" i.e. 5 dataframes in a second
cutoff=1.3

df_lowpass=Lowpass.low_pass_filter(df_lowpass,"acc_y", fs, cutoff,order=5)
subset=df_lowpass[df_lowpass["set"]==45] #subset containing only the rows where the value in the column "set" is equal to 45.
print(subset["label"][0])
df_lowpass.info()
 
#you can clearly see the difference between raw data and buterworth filter data the butterworth filter graph is more polished and very clear

for col in predictor_columns:
    df_lowpass=Lowpass.low_pass_filter(df_lowpass,col,fs,cutoff,order=5) #this function will return data table with one extra column which will have noiseless data
    df_lowpass[col]=df_lowpass[col+"_lowpass"] #we have replaced the actual column of dataframe with the return datatable of low_pass_filter() function
    del df_lowpass[col+"_lowpass"] #this will delete the extra column of lowpass
    

# --------------------------------------------------------------
# Principal component analysis PCA
# --------------------------------------------------------------
df_pca=df_lowpass.copy()

pca=PrincipalComponentAnalysis()

pc_values=pca.determine_pc_explained_variance(df_pca, predictor_columns)

plt.figure(figsize=(10,10))
plt.plot(range(1,len(predictor_columns)+1),pc_values)
plt.xlabel("principal component number")
plt.ylabel("explained variance")
plt.show()

df_pca=pca.apply_pca(df_pca,predictor_columns,3)
subset=df_pca[df_pca["set"]==35]
subset[["pca_1","pca_2","pca_3"]].plot()

# --------------------------------------------------------------
# Sum of squares attributes
# --------------------------------------------------------------
df_squared=df_pca.copy()

acc_r=df_squared["acc_x"] **2 + df_squared["acc_y"] **2 + df_squared["acc_z"] **2
gyr_r=df_squared["gyr_x"] **2 + df_squared["gyr_y"] **2 + df_squared["gyr_z"] **2

df_squared["acc_r"]=np.sqrt(acc_r)
df_squared["gyr_r"]=np.sqrt(gyr_r)

subset = df_squared[df_squared["set"]==14]

subset[["acc_r", "gyr_r"]].plot(subplots=True)

# --------------------------------------------------------------
# Temporal abstraction
# --------------------------------------------------------------

df_temporal=df_squared.copy()
NumAbs=NumericalAbstraction()

predictor_columns=predictor_columns + ["acc_r","gyr_r"]

ws=int(1000/200) #i.e 5 data entries in a second

# for col in predictor_columns:
#     df_temporal=NumAbs.abstract_numerical(df_temporal,[col],ws,"mean")
#     df_temporal=NumAbs.abstract_numerical(df_temporal,[col],ws,"std")

df_temporal=NumAbs.abstract_numerical(df_temporal,predictor_columns,ws,"mean")
df_temporal=NumAbs.abstract_numerical(df_temporal,predictor_columns,ws,"std")

df_temporal_list=[]
for s in df_temporal["set"].unique():
    subset=df_temporal[df_temporal["set"]==s].copy()
    for col in predictor_columns:
        subset=NumAbs.abstract_numerical(subset,[col],ws,"mean")   
        subset=NumAbs.abstract_numerical(subset,[col],ws,"std")   
    df_temporal_list.append(subset)

df_temporal=pd.concat(df_temporal_list)      

subset[["acc_y","acc_y_temp_mean_ws_5","acc_y_temp_std_ws_5"]].plot() 
subset[["gyr_y","gyr_y_temp_mean_ws_5","gyr_y_temp_std_ws_5"]].plot() 
    
# ------------------------------------------------------------------
# Frequency features
# ------------------------------------------------------------------
df_freq=df_temporal.copy().reset_index()
FreqAbs=FourierTransformation()

fs=int(1000/200) #5 samples in a second
ws=int(2800/200) #window size we gonna set to average length for a repitition i.e 2.8sec

df_freq=FreqAbs.abstract_frequency(df_freq,["acc_y"],ws,fs)

subset=df_freq[df_freq["set"]==15]
subset.info()
subset[["acc_y"]].plot()
subset[
    [
        "acc_y_max_freq",
        "acc_y_freq_weighted",
        "acc_y_pse",
        "acc_y_freq_1.429_Hz_ws_14",
        "acc_y_freq_2.5_Hz_ws_14"
    ]
].plot()

df_freq_list=[]
for s in df_freq["set"].unique():
    print(f"Applying Fourier transformation to set {s}")
    subset=df_freq[df_freq["set"]==s].reset_index(drop=True).copy()
    subset=FreqAbs.abstract_frequency(subset,predictor_columns,ws,fs)
    df_freq_list.append(subset)
    
df_freq=pd.concat(df_freq_list).set_index("epoch (ms)", drop=True)    

# --------------------------------------------------------------
# Dealing with overlapping windows
# --------------------------------------------------------------
df_freq=df_freq.dropna()

df_freq=df_freq.iloc[::2] #This will contain alternate rows in a data
# --------------------------------------------------------------
# Clustering
# --------------------------------------------------------------
df_cluster=df_freq.copy()

cluster_columns=["acc_x","acc_y","acc_z"]
k_values=range(2,10)
inertias=[]

for k in k_values:
    subset=df_cluster[cluster_columns]
    kmeans=KMeans(n_clusters=k,n_init=20,random_state=0)
    cluster_labels=kmeans.fit_predict(subset)
    inertias.append(kmeans.inertia_)

plt.figure(figsize=(10,10))
plt.plot(k_values,inertias)
plt.xlabel("k")
plt.ylabel("Sum of squared distances")
plt.show()

kmeans=KMeans(n_clusters=5,n_init=20,random_state=0)
subset=df_cluster[cluster_columns]
df_cluster["cluster"]=kmeans.fit_predict(subset)
df_cluster.info()
#Plot clusters
fig=plt.figure(figsize=(15,15))
ax=fig.add_subplot(projection="3d")
for l in df_cluster["label"].unique():
    subset=df_cluster[df_cluster["label"]==l]
    ax.scatter(subset["acc_x"],subset["acc_y"],subset["acc_z"], label=l)
ax.set_xlabel("X-axis")    
ax.set_ylabel("Y-axis")    
ax.set_zlabel("Z-axis")    
plt.legend()
plt.show()
# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------

df_cluster.to_pickle("../../data/interim/03_data_features.pkl")