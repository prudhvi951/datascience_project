# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load Dataset
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Display the first few rows
print(train_df.head())

# Data Exploration
# Check for missing values
missing_values = train_df.isnull().sum().sort_values(ascending=False)
missing_values = missing_values[missing_values > 0]
print(missing_values)

# Summary statistics
print(train_df.describe())

# Distribution of the target variable
sns.histplot(train_df['SalePrice'], kde=True)
plt.title('Distribution of SalePrice')
plt.show()

# Correlation matrix
corr_matrix = train_df.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Data Preprocessing and Feature Engineering
# Select features and target
X = train_df.drop(['SalePrice', 'Id'], axis=1)
y = train_df['SalePrice']

# List of numerical and categorical columns
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = X.select_dtypes(include=['object']).columns

# Preprocessing pipelines for numerical and categorical data
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ]
)

# Model Training
# Split the data
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=0)

# Create pipelines for models
model_pipeline_lr = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', LinearRegression())
])

model_pipeline_rf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(n_estimators=100, random_state=0))
])

# Train and evaluate Linear Regression
model_pipeline_lr.fit(X_train, y_train)
y_pred_lr = model_pipeline_lr.predict(X_valid)
rmse_lr = np.sqrt(mean_squared_error(y_valid, y_pred_lr))
print(f'Linear Regression RMSE: {rmse_lr}')

# Train and evaluate Random Forest
model_pipeline_rf.fit(X_train, y_train)
y_pred_rf = model_pipeline_rf.predict(X_valid)
rmse_rf = np.sqrt(mean_squared_error(y_valid, y_pred_rf))
print(f'Random Forest RMSE: {rmse_rf}')

# Prediction on Test Set and Submission
# Preprocess and predict on the test set using the better model
X_test = test_df.drop(['Id'], axis=1)
predictions = model_pipeline_rf.predict(X_test)

# Prepare submission file
submission = pd.DataFrame({
    'Id': test_df['Id'],
    'SalePrice': predictions
})
submission.to_csv('submission.csv', index=False)
