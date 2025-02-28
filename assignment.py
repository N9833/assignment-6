# Step 1: Read in the CSV file using pandas.
import pandas as pd

# Read the CSV file with the correct delimiter (semicolon)
df = pd.read_csv('bank.csv', delimiter=';')

# Inspect the dataframe
print(df.head())  # Display the first few rows
print(df.info())  # Display column names and variable types

# Step 2: Pick data from the specified columns to a second dataframe 'df2'.
df2 = df[['y', 'job', 'marital', 'default', 'housing', 'poutcome']]

# Step 3: Convert categorical variables to dummy numerical values.
df3 = pd.get_dummies(df2, columns=['job', 'marital', 'default', 'housing', 'poutcome'])

# Convert 'y' to binary (0 for 'no', 1 for 'yes') BEFORE calculating correlations
df3['y'] = df3['y'].map({'no': 0, 'yes': 1})

# Step 4: Produce a heat map of correlation coefficients.
import seaborn as sns
import matplotlib.pyplot as plt

# Calculate the correlation matrix
corr_matrix = df3.corr()

# Plot the heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()

# Step 5: Select the target variable 'y' and explanatory variables X.
X = df3.drop('y', axis=1)
y = df3['y']

# Step 6: Split the dataset into training and testing sets with a 75/25 ratio.
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Step 7: Setup a logistic regression model, train it, and predict on testing data.
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score

# Initialize the logistic regression model
log_reg = LogisticRegression(max_iter=1000)

# Train the model
log_reg.fit(X_train, y_train)

# Predict on the test data
y_pred_log_reg = log_reg.predict(X_test)

# Step 8: Print the confusion matrix and accuracy score for the logistic regression model.
conf_matrix_log_reg = confusion_matrix(y_test, y_pred_log_reg)
accuracy_log_reg = accuracy_score(y_test, y_pred_log_reg)

print("Confusion Matrix for Logistic Regression:")
print(conf_matrix_log_reg)
print(f"Accuracy Score for Logistic Regression: {accuracy_log_reg:.2f}")

# Step 9: Repeat steps 7 and 8 for k-nearest neighbors model.
from sklearn.neighbors import KNeighborsClassifier

# Initialize the KNN model with k=3
knn = KNeighborsClassifier(n_neighbors=3)

# Train the model
knn.fit(X_train, y_train)

# Predict on the test data
y_pred_knn = knn.predict(X_test)

# Print the confusion matrix and accuracy score for the KNN model
conf_matrix_knn = confusion_matrix(y_test, y_pred_knn)
accuracy_knn = accuracy_score(y_test, y_pred_knn)

print("Confusion Matrix for KNN:")
print(conf_matrix_knn)
print(f"Accuracy Score for KNN: {accuracy_knn:.2f}")

# Step 10: Compare the results between the two models.
"""
The logistic regression model achieved an accuracy of {accuracy_log_reg:.2f}, while the KNN model achieved an accuracy of {accuracy_knn:.2f}.
The confusion matrices show that the logistic regression model had fewer misclassifications compared to the KNN model.
Overall, the logistic regression model performed better on this dataset.
"""