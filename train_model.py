import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Read dataset
data = pd.read_csv("student_data.csv")

# Input variables
X = data[
    [
        "attendance",
        "study_hours",
        "previous_score",
        "assignments",
        "internal_marks"
    ]
]

# Output variable
y = data["final_score"]

# Divide dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create AI model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

# Train model
model.fit(X_train, y_train)

# Test model
predictions = model.predict(X_test)

print("Mean Absolute Error:",
      mean_absolute_error(y_test, predictions))

print("R2 Score:",
      r2_score(y_test, predictions))

# Save trained model
with open("student_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model successfully trained!")
print("student_model.pkl created.")
