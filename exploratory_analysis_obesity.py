import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load the CSV data into a Pandas DataFrame
# Replace 'your_data.csv' with the actual filename of your CSV file
df = pd.read_csv('obesity.csv')

# Step 2: Check for missing values
null_counts = df.isnull().sum()

# Step 3: Visualize the missing values (optional)
plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Values Heatmap')
plt.show()

# Step 4: Explore the dataset
# Summary statistics
print("Summary Statistics:")
print(df.describe())

# Data distribution (example for numeric columns)
numeric_columns = df.select_dtypes(include=['number']).columns
for column in numeric_columns:
    plt.figure(figsize=(8, 6))
    sns.histplot(df[column], kde=True)
    plt.title(f'Distribution of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.show()

# Categorical data analysis (example for categorical columns)
categorical_columns = df.select_dtypes(include=['object']).columns
for column in categorical_columns:
    plt.figure(figsize=(8, 6))
    sns.countplot(data=df, x=column)
    plt.title(f'Countplot of {column}')
    plt.xlabel(column)
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.show()

# Correlation analysis (example for numeric columns)
correlation_matrix = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=.5)
plt.title('Correlation Heatmap')
plt.show()
