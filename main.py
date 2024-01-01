import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

# Load the dataset
csv_file_path = r'E:\Amir kabir university\Term5\hooshh\creditcard.csv'
df = pd.read_csv(csv_file_path)

# Remove duplicate rows
df = df.drop_duplicates()

# Split the dataset into features (X) and target variable (y)
X = df.drop('Class', axis=1)  # Adjust 'target_column_name' to your actual target column
y = df['Class']

# Split into training and testing sets using a fixed random state
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Normalize the features using StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Implement Logistic Regression model
logistic_model = LogisticRegression()
logistic_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = logistic_model.predict(X_test)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Additional evaluation metrics
print('\nClassification Report:\n', classification_report(y_test, y_pred))
print('\nConfusion Matrix:\n', confusion_matrix(y_test, y_pred))

# Calculate and print F1 score
f1 = f1_score(y_test, y_pred)
print(f'F1 Score: {f1:.2f}')


