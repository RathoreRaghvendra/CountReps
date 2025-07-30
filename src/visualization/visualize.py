import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from Ipython.display import display

# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------
df=pd.read_pickle("../../data/interim/01_data_processed.pkl")
df.info()
# --------------------------------------------------------------
# Plot single columns
# --------------------------------------------------------------
set_df=df[df["set"]==1]
plt.plot(set_df["acc_y"].reset_index(drop=True))
# --------------------------------------------------------------
# Plot all exercises
# --------------------------------------------------------------

for label in df["label"].unique():
    subset=df[df["label"]==label]
    fig, ax=plt.subplots() #plt.subplots() is a function that 
    #returns a tuple containing a figure and axes object(s). 
    # Thus when using fig, ax = plt.subplots() you unpack this tuple
    # into the variables fig and ax. Having fig is useful if you want 
    # to change figure-level attributes or save the figure as an image file later (e.g. with fig.savefig('yourfilename.png')). 
    # You certainly don't have to use the returned figure object but many people do use it later so it's common to see. Also, all axes objects (the objects that have plotting methods), have a parent figure object anyway
    plt.plot(subset["acc_y"].reset_index(drop=True),label=label)#it will plot all the data like acc_y in the graph 
    plt.legend() #It  gives a way to label and differentiate between multiple plots in the same figure
    plt.show() #It's used to display all figures
    
for label in df["label"].unique():
    subset=df[df["label"]==label]
    fig, ax=plt.subplots() 
    plt.plot(subset[:100]["acc_y"].reset_index(drop=True),label=label)
    plt.legend() 
    plt.show() 
    

# --------------------------------------------------------------
# Adjust plot settings
# --------------------------------------------------------------
mpl.style.use("seaborn-v0_8-deep")
mpl.rcParams["figure.figsize"]=(20,5) #it's determing height and width of graph
mpl.rcParams["figure.dpi"]=100

# --------------------------------------------------------------
# Compare medium vs. heavy sets
# --------------------------------------------------------------

category_df = df[df["label"]=='squat'] #this is also method for selection

category_df = df.query("label == 'squat'").query("participant == 'A'").reset_index() #this is the query method to acces any column
fig, ax = plt.subplots()
category_df.groupby(["category"])["acc_y"].plot()
ax.set_ylabel("acc_y")
ax.set_xlabel("samples")
plt.legend()


# --------------------------------------------------------------
# Compare participants
# --------------------------------------------------------------

participant_df=df.query("label =='bench'").sort_values("participant").reset_index()
fig, ax= plt.subplots(figsize=(25,10))
participant_df.groupby(["participant"])["acc_y"].plot()
ax.set_ylabel("acc_y")
ax.set_xlabel("acc_x")
ax.set_title("bench")
plt.legend()

# --------------------------------------------------------------
# Plot multiple axis
# --------------------------------------------------------------
label="squat"
participant="A"
all_axis_df=df.query(f"label == '{label}'").query(f"participant == '{participant}'").reset_index()

fig, ax=plt.subplots()
all_axis_df[["acc_x","acc_y", "acc_z"]].plot(ax=ax)
ax.set_ylabel("acc_y")
ax.set_xlabel("samples")
plt.legend()


# --------------------------------------------------------------
# Create a loop to plot all combinations per sensor
# --------------------------------------------------------------
labels=df["label"].unique()
participants=df["participant"].unique()
for label in labels:
    for participant in participants:
        all_axis_df=df.query(f"label == '{label}'").query(f"participant == '{participant}'").reset_index()
        
        if len(all_axis_df)>0:            
            fig, ax=plt.subplots()
            all_axis_df[["acc_x","acc_y","acc_z"]].plot(ax=ax)
            ax.set_ylabel("acc_y")
            ax.set_xlabel("samples")
            plt.title(f"{label} ({participant})".title())
            plt.legend()

for label in labels:
    for participant in participants:
        all_axis_df=df.query(f"label == '{label}'").query(f"participant == '{participant}'").reset_index()
        
        if len(all_axis_df)>0:            
            fig, ax=plt.subplots()
            all_axis_df[["gyr_x","gyr_y","gyr_z"]].plot(ax=ax)
            ax.set_ylabel("gyr_y")
            ax.set_xlabel("samples")
            plt.title(f"{label} ({participant})".title())
            plt.legend()            
            
                        
# --------------------------------------------------------------
# Combine plots in one figure
# --------------------------------------------------------------
label="row"
participant="A"
combined_plot_df=df.query(f"label == '{label}'").query(f"participant == '{participant}'").reset_index()
fig,ax=plt.subplots(nrows=2,sharex=True,figsize=(30,15))
plt.rcParams.update({'font.size': 22})
combined_plot_df[["acc_x","acc_y","acc_z"]].plot(ax=ax[0])
combined_plot_df[["gyr_x","gyr_y","gyr_z"]].plot(ax=ax[1])

ax[0].legend(loc="upper center", bbox_to_anchor=(0.5,1.15), ncol=3, fancybox=True, shadow= True, fontsize="large")
ax[1].legend(loc="upper center", bbox_to_anchor=(0.5,1.15), ncol=3, fancybox=True, shadow= True, fontsize='large')
ax[1].set_xlabel("samples")

# ---------------------------------------------------------------
# Loop over all combinations and export for both sensors
# ---------------------------------------------------------------


labels=df["label"].unique()
participants=df["participant"].unique()
for label in labels:
    for participant in participants:
        combined_plot_df=df.query(f"label == '{label}'").query(f"participant == '{participant}'").reset_index()
        if len(combined_plot_df)>0:            
            fig, ax=plt.subplots(nrows=2,sharex=True,figsize=(20,10))
            combined_plot_df[["acc_x","acc_y","acc_z"]].plot(ax=ax[0])
            combined_plot_df[["gyr_x","gyr_y","gyr_z"]].plot(ax=ax[1])
                           
            ax[0].legend(loc="upper center", bbox_to_anchor=(0.5,1.15), ncol=3, fancybox=True, shadow= True, fontsize='large')
            ax[1].legend(loc="upper center", bbox_to_anchor=(0.5,1.15), ncol=3, fancybox=True, shadow= True, fontsize='large')
            ax[1].set_xlabel("samples")
            plt.title(f"{label} ({participant})".title(),loc='left')
            plt.show()
            break
            
combined_plot_df.to_csv("../../data/interim/03_csv_data_processed.csv") # we have used pickle bcoz it exports datasets in a serialized format 