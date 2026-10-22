from flask import Flask, request, jsonify
import joblib
import numpy as np

# Load the trained model
model = joblib.load('Breast_cancer.pkl')

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the JSON data from the request
        data = request.get_json()

        # Check if data is in the correct format
        if not isinstance(data, list):
            return jsonify({"error": "Input data must be a list of features"}), 400

        # Convert the input data to a NumPy array and reshape it
        new_data = np.array(data).reshape(1, -1)

        # Make predictions
        prediction = model.predict(new_data)[0]

        # Return the result as JSON
        result = {
            "Predicted target variable": int(prediction)
        }
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
