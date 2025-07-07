from flask import Flask, render_template, request, session
import numpy as np
import joblib
from tensorflow.keras.models import load_model

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Required for session usage

# Load model and scaler
model = load_model("lstm_model1.h5")
scaler = joblib.load("scaler.pkl")

# Sample 82-value input (float values as example)
sample_input = ",".join([str(round(np.random.uniform(0, 1), 4)) for _ in range(82)])

@app.route("/", methods=["GET", "POST"])
def predict():
    prediction = None
    if request.method == "POST":
        try:
            features = request.form["features"]
            values = np.array([float(x.strip()) for x in features.split(",")])

            if len(values) != 82:
                prediction = "Error: Input must contain exactly 82 features."
            else:
                scaled_input = scaler.transform([values])
                reshaped_input = scaled_input.reshape((1, 1, 82))
                pred = model.predict(reshaped_input)
                prediction = round(float(pred[0][0]), 2)

                # Save to session history
                if "history" not in session:
                    session["history"] = []
                session["history"].append(f"{prediction}")
        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template(
        "index.html",
        prediction=prediction,
        sample_input=sample_input,
        history=session.get("history", [])
    )

if __name__ == "__main__":
    app.run(debug=True)


