import pandas as pd
import glob

# --------------------------------------------------------------
# Read single CSV file
# --------------------------------------------------------------
single_file_acc = pd.read_csv(
    "../../data/raw/MetaMotion/A-bench-heavy2-rpe8_MetaWear_2019-01-11T16.10.08.270_C42732BE255C_Accelerometer_12.500Hz_1.4.4.csv"
)
single_file_gyr = pd.read_csv(
    "../../data/raw/MetaMotion/A-bench-heavy2-rpe8_MetaWear_2019-01-11T16.10.08.270_C42732BE255C_Gyroscope_25.000Hz_1.4.4.csv"
)
# --------------------------------------------------------------
# List all data in data/raw/MetaMotion
# --------------------------------------------------------------

files = glob.glob("../../data/raw/MetaMotion/*.csv")
len(files)

# --------------------------------------------------------------
# Extract features from filename
# --------------------------------------------------------------

datapath="../../data/raw/MetaMotion"
f=files[0]
participant =f.split("-")[0].replace(datapath+"\\","")
label = f.split('-')[1]
category =f.split("-")[2].rstrip("123") 

df=pd.read_csv(f)

df["participant"]= participant
df["label"]= label
df["category"]= category

# --------------------------------------------------------------
# Read all files
# --------------------------------------------------------------

acc_df=pd.DataFrame()
gyr_df=pd.DataFrame()

acc_set=1
gyr_set=1

for f in files:
    participant =f.split("-")[0].replace(datapath+"\\","")
    label = f.split('-')[1]
    category =f.split("-")[2].rstrip("123").rstrip("_MetaWear_2019")
    
    df=pd.read_csv(f)
    
    df["participant"]= participant
    df["label"]= label
    df["category"]= category
        
    if "Accelerometer" in f:
        df["set"]=acc_set
        acc_set+=1
        acc_df=pd.concat([acc_df,df])
        
    if "Gyroscope" in f:
        df["set"]=gyr_set
        gyr_set+=1
        gyr_df=pd.concat([gyr_df,df])    

# --------------------------------------------------------------
# Working with datetimes
# --------------------------------------------------------------

acc_df.info()
pd.to_datetime(df["epoch (ms)"], unit="ms") #before this panda was not aware that epoch is a time bcoz its data type is int64, so to perform operations we have to convert it into time
pd.to_datetime(df["time (01:00)"])
pd.to_datetime(df["time (01:00)"]).dt.weekday

acc_df.index= pd.to_datetime(acc_df["epoch (ms)"], unit="ms") #by this unix time is converted to daytime for accelerometer data
gyr_df.index= pd.to_datetime(gyr_df["epoch (ms)"], unit="ms")

del acc_df["epoch (ms)"] #by deleting this we will delete an extra column of epoch (ms) in accelerometer df
del acc_df["time (01:00)"]
del acc_df["elapsed (s)"]

del gyr_df["epoch (ms)"]
del gyr_df["time (01:00)"]
del gyr_df["elapsed (s)"]



# --------------------------------------------------------------
# Turn into function
# --------------------------------------------------------------

files = glob.glob("../../data/raw/MetaMotion/*.csv")

def read_data_from_files(files):
    acc_df=pd.DataFrame()
    gyr_df=pd.DataFrame()
    acc_set=1
    gyr_set=1

    for f in files:
        participant =f.split("-")[0].replace(datapath+"\\","")
        label = f.split('-')[1]
        category =f.split("-")[2].rstrip("123").rstrip("_MetaWear_2019")
        
        df=pd.read_csv(f)
        
        df["participant"]= participant
        df["label"]= label
        df["category"]= category

        if "Accelerometer" in f:
            df["set"]=acc_set
            acc_set+=1
            acc_df=pd.concat([acc_df,df])
        
        if "Gyroscope" in f:
            df["set"]=gyr_set
            gyr_set+=1
            gyr_df=pd.concat([gyr_df,df])
            
    acc_df.index=pd.to_datetime(acc_df["epoch(ms)"], unit="ms")     
    gyr_df.index=pd.to_datetime(gyr_df["epoch(ms)"], unit="ms")  
    
    del acc_df["epoch(ms)"]
    del acc_df["time(01:00)"]
    del acc_df["elapsed(s)"]

    del gyr_df["epoch(ms)"]
    del gyr_df["time(01:00)"]
    del gyr_df["elapsed(s)"] 
    
    return acc_df, gyr_df

acc_df, gyr_df = read_data_from_files(files)

# --------------------------------------------------------------
# Merging datasets
# --------------------------------------------------------------

#axis=1 is bcoz we have to concat datasets column wise if axis=0 then it'll concat row wise

data_merged = pd.concat([acc_df.iloc[:,:3],gyr_df], axis=1) #iloc is used to take upto 3 starting columns of acc_df

#Renaming columns
data_merged.columns=[
    "acc_x",
    "acc_y",
    "acc_z",
    "gyr_x",
    "gyr_y",
    "gyr_z",
    "participant",  
    "label",
    "category",
    "set"
]

data_merged.head(50)


# --------------------------------------------------------------
# Resample data (frequency conversion)
# --------------------------------------------------------------

# Accelerometer:    12.500HZ
# Gyroscope:        25.000Hz

#Below we have created a dictionary which it used to assign.mean is for numerical value & last is for categorical value

sampling={
    "acc_x": "mean",
    "acc_y": "mean",
    "acc_z": "mean",
    "gyr_x": "mean",
    "gyr_y": "mean",
    "gyr_z": "mean",
    "participant": "last",
    "label": "last",
    "category": "last",
    "set": "last"
}

data_merged[:1000].resample(rule="200ms").apply(sampling) #the rule is specified in panda libraries, we have used 200ms which means 5 dataframes in a second

#split by day bcoz whole the data is recorded in a week
days=[g for n, g in data_merged.groupby(pd.Grouper(freq="D"))] #days is list
days[0]
data_resampled=pd.concat([df.resample(rule="200ms").apply(sampling).dropna() for df in days]) #we will loop all over days as df  and will apply sampling rule 
data_resampled.info()
data_resampled["set"]=data_resampled["set"].astype("int")

# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------
data_resampled.to_pickle("../../data/interim/01_data_processed.pkl") # we have used pickle bcoz it exports datasets in a serialized format 
data_resampled.to_csv("../../data/interim/01_csv__data_processed.pkl") # we have used pickle bcoz it exports datasets in a serialized format 