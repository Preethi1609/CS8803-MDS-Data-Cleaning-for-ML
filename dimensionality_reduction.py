import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

# Create a DataFrame from your dataset
data = pd.read_csv('obesity.csv')

# Handle missing values by imputing the mean for numeric columns and the most frequent value for categorical columns
numeric_columns = data.select_dtypes(include=['float64']).columns
categorical_columns = data.select_dtypes(exclude=['float64']).columns

numeric_imputer = SimpleImputer(strategy='mean')
categorical_imputer = SimpleImputer(strategy='most_frequent')

data[numeric_columns] = numeric_imputer.fit_transform(data[numeric_columns])
data[categorical_columns] = categorical_imputer.fit_transform(data[categorical_columns])

# Split the dataset into features and labels
X = data.iloc[:, :-1]  # Features (all columns except the last one)
y = data.iloc[:, -1]   # Labels (the last column)

# Convert categorical variables into numerical using one-hot encoding
X = pd.get_dummies(X, drop_first=True)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize a RandomForestClassifier (you can choose a different classifier if you prefer)
clf = RandomForestClassifier(random_state=42)

# Fit the classifier to the training data
clf.fit(X_train, y_train)

# Get feature importances from the classifier
feature_importances = clf.feature_importances_

# Create a DataFrame to store feature names and their importances
importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})

# Sort the features by importance in descending order
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Print the feature importances
print(importance_df)

# You can set a threshold for feature importance and drop features with importance below the threshold
threshold = 0.01  # Adjust this threshold as needed
least_significant_features = importance_df[importance_df['Importance'] < threshold]['Feature']

# Drop the least significant columns from the original dataset
reduced_data = data.drop(least_significant_features, axis=1)

# Print the reduced dataset
print(reduced_data)
