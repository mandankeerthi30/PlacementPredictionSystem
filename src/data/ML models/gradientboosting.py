import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv(
    r"C:\Users\hp\PycharmProjects\Placementpredictionsystem\data\placement_data (1).csv"
)

# Input and target
x = df[['CGPA']]
y = df['Salary Package']

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.2,
    random_state=42
)

# Create and train model
model = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)

model.fit(x_train, y_train)

# Prediction
y_pred = model.predict(x_test)

# Evaluation
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Display metrics
print("----- Gradient Boosting Performance -----")
print(f"MSE  : {mse:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R2   : {r2:.2f}")

# Feature importance
print(f"CGPA Importance : {model.feature_importances_[0]:.2f}")

# User input
my_cgpa = float(input("Enter your CGPA: "))

predicted_salary = model.predict([[my_cgpa]])

print(f"Your CGPA        : {my_cgpa:.2f}")
print(f"Predicted Salary : {predicted_salary[0]:.2f} LPA")