
from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load trained ML model
with open("student_model.pkl", "rb") as file:
    model = pickle.load(file)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        attendance = float(request.form["attendance"])
        study_hours = float(request.form["study_hours"])
        previous_score = float(request.form["previous_score"])
        assignments = float(request.form["assignments"])
        internal_marks = float(request.form["internal_marks"])

        # Create input for ML model
        input_data = pd.DataFrame(
            [[attendance, study_hours, previous_score,
              assignments, internal_marks]],
            columns=[
                "attendance",
                "study_hours",
                "previous_score",
                "assignments",
                "internal_marks"
            ]
        )

        # Prediction
        prediction = model.predict(input_data)[0]
        prediction = max(0, min(100, prediction))

        # Performance category
        if prediction >= 85:
            performance = "Excellent"
        elif prediction >= 70:
            performance = "Good"
        elif prediction >= 50:
            performance = "Average"
        else:
            performance = "Needs Improvement"

        # Risk category
        if prediction >= 70:
            risk = "Low Risk"
        elif prediction >= 50:
            risk = "Medium Risk"
        else:
            risk = "High Risk"

        return render_template(
            "result.html",
            score=round(prediction, 2),
            performance=performance,
            risk=risk
        )

    except Exception:
        return render_template(
            "result.html",
            error="Please enter valid values."
        )


if __name__ == "__main__":
    app.run(debug=True)
