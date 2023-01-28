from sklearn.utils import resample
from sklearn.preprocessing import LabelBinarizer
import pandas as pd


#load data
df = pd.read_csv("/Users/mauropazmino/Documents/chapter_2/data/raw/DRIFT_Legs2020.csv")

# Asses group count
class_counts = df.groupby("Age").size()
class_counts

# balance groups
class_counts = df.groupby("Age").size()
df_majority = df[df.Age=="10D"]
df_minority = df[df.Age=="03D"]
 
# Downsample majority class
df_majority_downsampled = resample(df_majority, 
                                    replace=False,    # sample without replacement
                                    n_samples=min(class_counts), # to match minority class
                                    random_state=123) # reproducible results
 
# Combine minority class with downsampled majority class
df_balanced = pd.concat([df_majority_downsampled, df_minority])


# Seperate labels and features 
X = df_balanced.loc[:,"1800":"600"]
y = df_balanced.loc[:,"Age"]

# export

X.to_csv("/Users/mauro/Documents/Github/chapter_2/chapter_2/data/processed/X_age.csv", index=False)
y.to_csv("/Users/mauro/Documents/Github/chapter_2/chapter_2/data/processed/y_age.csv",index=False)


# DATA PREPARATION FOR SPECIES
# Asses group count
class_counts = df.groupby("Species").size()
class_counts

# balance groups
df_majority = df[df.Species=="AK"]
df_minority = df[df.Species=="AC"]

df_majority_downsampled = resample(df_majority, 
                                    replace=False,    # sample without replacement
                                    n_samples=min(class_counts), # to match minority class
                                    random_state=123) # reproducible results
 
# Combine minority class with downsampled majority class
df_balanced = pd.concat([df_majority_downsampled, df_minority])

X = df_balanced.loc[:,"1800":"600"]
y = df_balanced.loc[:,"Species"]

X.to_csv("data/processed/X_species.csv", index=False)
y.to_csv("data/processed/y_species.csv",index=False)


# Data preparation for strain and insecticide resistance
df = pd.read_csv("data/raw/DRIFT_InsRes2.csv")

class_counts = df.groupby("Status").size()
class_counts

# balance groups
df_majority = df[df.Status=="Suceptible"]
df_minority = df[df.Status=="Resistant"]

df_majority_downsampled = resample(df_majority, 
                                    replace=False,    # sample without replacement
                                    n_samples=min(class_counts), # to match minority class
                                    random_state=123) # reproducible results
 
# Combine minority class with downsampled majority class
df_balanced = pd.concat([df_majority_downsampled, df_minority])

X = df_balanced.loc[:,"1800":"600"]
y = df_balanced.loc[:,"Status"]

X.to_csv("data/processed/X_status.csv", index=False)
y.to_csv("data/processed/y_status.csv",index=False)

# Strains

class_counts = df.groupby("Species").size()
class_counts

df_majority = df[df.Species=="AT"]
df_minority = df[df.Species=="AK"]
df_normal = df[df.Species=="AN"]

df_majority_downsampled = resample(df_majority, 
                                    replace=False,    # sample without replacement
                                    n_samples=min(class_counts), # to match minority class
                                    random_state=123) # reproducible results

dfs = [df_minority, df_majority_downsampled, df_normal]

df_balanced = pd.concat(dfs)

X = df_balanced.loc[:,"1800":"600"]
y = df_balanced.loc[:,"Species"]

X.to_csv("data/processed/X_strains.csv", index=False)
y.to_csv("data/processed/y_strains.csv",index=False)

