# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# --- Step 1: Import the dataset and explore basic info ---
print("Step 1: Loading and Exploring the Dataset")
# Load the dataset from a CSV file. Make sure 'Titanic-Dataset.csv' is in the same directory as the script.
try:
    df = pd.read_csv('Titanic-Dataset.csv')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: 'Titanic-Dataset.csv' not found. Please check the file path.")
    exit()

# Display basic information about the dataframe
print("\nDataFrame Info:")
df.info()

# Display the first 5 rows of the dataframe
print("\nDataFrame Head:")
print(df.head())


# --- Step 2: Handle missing values ---
print("\nStep 2: Handling Missing Values")
print("\nMissing values before handling:")
print(df.isnull().sum())

# Fill missing 'Age' values with the median. Median is less sensitive to outliers than the mean.
df['Age'].fillna(df['Age'].median(), inplace=True)

# Fill missing 'Embarked' values with the mode (the most frequent value).
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# The 'Cabin' column has too many missing values to be useful, so we drop it.
df.drop('Cabin', axis=1, inplace=True)

print("\nMissing values after handling:")
print(df.isnull().sum())


# --- Step 3: Convert categorical features into numerical ---
print("\nStep 3: Encoding Categorical Features")
# Use one-hot encoding to convert 'Sex' and 'Embarked' into numerical format.
# drop_first=True avoids multicollinearity by removing one of the encoded columns.
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)
print("Categorical features 'Sex' and 'Embarked' have been encoded.")


# --- Step 4: Normalize/Standardize numerical features ---
print("\nStep 4: Normalizing Numerical Features")
# We'll standardize 'Age' and 'Fare' to have a mean of 0 and a standard deviation of 1.
# This is important for many ML algorithms.
numerical_features = ['Age', 'Fare']
scaler = StandardScaler()
df[numerical_features] = scaler.fit_transform(df[numerical_features])
print("'Age' and 'Fare' columns have been standardized.")
print("\nDataFrame after encoding and normalization:")
print(df.head())


# --- Step 5: Visualize and remove outliers ---
print("\nStep 5: Visualizing and Removing Outliers")
# Create boxplots to visualize the distribution and identify outliers in 'Age' and 'Fare'.
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.boxplot(y=df['Age'])
plt.title('Boxplot of Standardized Age')

plt.subplot(1, 2, 2)
sns.boxplot(y=df['Fare'])
plt.title('Boxplot of Standardized Fare')
plt.tight_layout()
# Save the plot to a file
plt.savefig('outlier_boxplots.png')
print("Generated 'outlier_boxplots.png' to show data distribution.")

# Define a function to remove outliers using the Interquartile Range (IQR) method.
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Filter the dataframe to keep only the rows within the bounds
    df_filtered = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return df_filtered

# Remove outliers from 'Age' and 'Fare'
df_no_outliers = remove_outliers(df, 'Age')
df_no_outliers = remove_outliers(df_no_outliers, 'Fare')

print(f"\nOriginal number of rows: {df.shape[0]}")
print(f"Number of rows after removing outliers: {df_no_outliers.shape[0]}")
print(f"Number of outliers removed: {df.shape[0] - df_no_outliers.shape[0]}")


# --- Save the Cleaned Data ---
# Save the final, cleaned dataframe to a new CSV file.
df_no_outliers.to_csv('Titanic-Dataset-Cleaned.csv', index=False)
print("\nSuccessfully saved the cleaned data to 'Titanic-Dataset-Cleaned.csv'")
