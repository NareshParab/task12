#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Raw data
df = pd.read_csv('Canada-2019.csv')

# Bar chart for Gender distribution
plt.figure(figsize=(16, 18))
sns.countplot(x="Age", data=df, palette="Set2")
plt.title("Age Distribution", fontsize=16)
plt.xlabel("Age", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.grid(axis='y',linestyle='--',alpha=0.7)
plt.show()

# Histogram for Age distribution
plt.figure(figsize=(16, 18))
sns.histplot(df["Age"], bins=10, kde=True, color="blue")
plt.title("Age Distribution", fontsize=16)
plt.xlabel("Age", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.show()


# # Performing data cleaning,preprocessing & visulaizing the relationships between different variables of dataset

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Load Titanic dataset 
df = pd.read_csv('titanic test.csv')
df


# In[3]:


# Check for missing values
print(df.isnull().sum())


# In[4]:


# Check for duplicates
print(f"Number of duplicates: {df.duplicated().sum()}")
df.drop_duplicates(inplace=True)


# In[5]:


# Convert categorical columns to category type
categorical_cols = ['Sex', 'Embarked', 'Age', 'Ticket', 'Cabin']
df[categorical_cols] = df[categorical_cols].astype('category')


# In[6]:


print(df.columns)
print(df[categorical_cols].isna().sum())
categorical_cols = ['Sex', 'Embarked', 'Ticket', 'Cabin']  # Exclude 'Age'
print(categorical_cols )
for col in categorical_cols:
    if col in df.columns:
        df[col] = df[col].astype('category')
# Summary of numerical variables
print(df.describe())

# Summary of categorical variables
print(df.describe(include='category'))


# In[7]:


# Distribution of Age # Plot the histogram
df = pd.read_csv('titanic test.csv')
df
if 'Age' in df.columns:
    plt.figure(figsize=(8, 5))
    sns.histplot(df['Age'], kde=True, bins=20, color='blue')
    plt.title("Age Distribution")
    plt.xlabel("Age")
    plt.ylabel("Frequency")
    plt.show()
else:
    print("Error: 'Age' column is missing in the dataset.")


# In[37]:


df = pd.read_csv('titanic raw.csv')
print(df)
# # Countplot for Survival
sns.countplot(x='Survived', data=df, palette='Set3')
plt.title("Survival Counts")
plt.xticks([0, 1], ['Did Not Survive', 'Survived'])
plt.show()


# In[9]:


df1 = pd.read_csv('titanic raw.csv')
df2 = pd.read_csv('titanic test.csv')
# Merge the datasets on 'PassengerId'
merged_df = pd.merge(df1, df2, on='PassengerId', suffixes=('_raw', '_test'))
merged_df


# In[38]:


sns.barplot(x='Sex', y='Survived', data=merged_df, palette='Set2')
plt.title("Survival Rate by Gender")
plt.show()


# In[43]:


# Age vs. Fare hue='Survived
custom_colors = {0: "red", 1: "green"} 
plt.figure(figsize=(8, 5))

sns.scatterplot(x='Age', y='Fare', hue='Survived',data=merged_df,palette=custom_colors)
plt.title("Age vs Fare with Survival Status")
plt.show()


# In[44]:


aggregated_df = merged_df.groupby(['Sex', 'Pclass']).agg({'Survived': 'mean'}).reset_index()
custom_colors = {0: "red", 1: "green"} 
sns.barplot(x='Pclass', y='Survived', hue='Sex', data=aggregated_df)
plt.title("Survival Rate by Passenger Class and Gender")
plt.show()

