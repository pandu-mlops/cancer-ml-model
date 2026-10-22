# Assignment 3: Dockerizing a breast cancer ML model

#### Task
Setting up a Docker container for a Flask app, which serves a machine learning model for predicting breast cancer. 

Tips: 
1. Setup the Project Structur
Create a project directory with the following structure:
```
breast_cancer_prediction/
│
├── Dockerfile
├── requirements.txt
├── predict.py
└── Breast_cancer.pkl

```
`Dockerfile`: Contains instructions to build the Docker image.
`requirements.txt`: Lists the Python dependencies.
`predict.py`: The Flask application script.
`Breast_cancer.pkl`: The serialized machine learning model.

2.  Create `requirements.txt`
This file specifies the dependencies needed for your application:
```
joblib
numpy
xgboost
flask
```
3. Create the `Dockerfile`
```
# Use the official Python image from the Docker Hub

# Set the working directory in the container

# Copy the current directory contents into the container at /app

# Install any needed packages specified in requirements.txt

# Expose the port the app runs on

# Run the Flask app
```

 4. Create the Flask App (`predict.py`)
 This script sets up a Flask app that loads a pre-trained model and provides an endpoint for predictions:
```
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

```
5. Build, Run & Push the Docker Image
6. You can use a separate Python script to send requests to the Flask application running inside the Docker container and test the predictions. Below is an example of how you might set up such a script, named `predict-test.py`. To generate predictions, make sure your Docker is running

```
import requests

# Define a sample data point for the breast cancer dataset
candidate = [842302, 17.99, 10.38, 122.8, 1001, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 
              0.07871, 1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904, 0.05373, 0.01587, 
              0.03003, 0.006193, 25.38, 17.33, 184.6, 2019, 0.1622, 0.6656, 0.7119, 0.2654, 
              0.4601, 0.1189]

# Define the URL for the Flask API
url = "http://localhost:9696/predict"

# Make the API call with the candidate data
response = requests.post(url=url, json=candidate)

# Check the response status and print the result
if response.status_code == 200:
    output = response.json()
    print(f'Prediction result: {output}')
else:
    print(f'Error in prediction, status code: {response.status_code}')

```
