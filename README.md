# Breast Cancer Prediction Using Flask and Docker

## 1. Project Overview

This project demonstrates how to deploy a machine learning model for breast cancer prediction using a Flask web application inside a Docker container.

The application accepts input features, sends them to a trained machine learning model, and displays the prediction through a web interface.

**Medical disclaimer:** This project is for educational purposes only. It is not a medical device and must not be used for diagnosis or treatment decisions.

## 2. Project Objectives

- Develop a Flask web application.
- Train and save a machine learning model.
- Create a web interface for predictions.
- Package the application using Docker.
- Run the application inside a Docker container.
- Learn basic containerization and model deployment.

## 3. Project Structure

```text
breast-cancer-docker/
├── app.py
├── train_model.py
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── model/
│   └── breast_cancer_model.pkl
└── templates/
    └── index.html
```

## 4. Technologies Used

- Python
- Flask
- Scikit-learn
- HTML
- Docker
- Gunicorn

## 5. Prerequisites

Install the following software:

- Python
- Docker Desktop
- Visual Studio Code (optional)
- A web browser

Verify Docker installation:

```bash
docker --version
docker info
```

## 6. Install Dependencies

Create `requirements.txt`:

```text
Flask
scikit-learn
joblib
numpy
gunicorn
```

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

## 7. Train the Machine Learning Model

Create `train_model.py`:

```python
from pathlib import Path

import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

data = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(
    data.data,
    data.target,
    test_size=0.2,
    random_state=42,
    stratify=data.target
)

model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ))
])

model.fit(X_train, y_train)

Path("model").mkdir(exist_ok=True)

joblib.dump(
    {
        "model": model,
        "feature_names": data.feature_names.tolist(),
        "target_names": data.target_names.tolist()
    },
    "model/breast_cancer_model.pkl"
)

print("Model saved successfully.")
print("Test accuracy:", model.score(X_test, y_test))
```

Run the training script:

```bash
python train_model.py
```

This creates the trained model file inside the `model` directory.

## 8. Create the Flask Application

Create `app.py`. The application should load the saved model, validate user inputs, and return a prediction.

Example starting point:

```python
from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

saved = joblib.load("model/breast_cancer_model.pkl")
model = saved["model"]
feature_names = saved["feature_names"]
target_names = saved["target_names"]

@app.route("/")
def home():
    return "Breast Cancer Prediction Application"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        values = [float(data[name]) for name in feature_names]

        result = int(model.predict(
            np.array(values).reshape(1, -1)
        )[0])

        return jsonify({
            "prediction": target_names[result]
        })

    except (TypeError, ValueError, KeyError):
        return jsonify({
            "error": "Provide valid numeric values for all features."
        }), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
```

This API expects all 30 features using the exact feature names returned by the dataset. The endpoint is an educational example, not a clinically validated prediction service.

## 9. Create the Dockerfile

Create a file named `Dockerfile`:

```dockerfile
FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY model/ model/
COPY templates/ templates/

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
```

### Explanation

- `FROM`: selects the Python base image.
- `WORKDIR`: sets the working directory inside the container.
- `COPY`: copies project files into the image.
- `RUN`: installs dependencies.
- `EXPOSE`: documents the application port.
- `CMD`: starts the Flask application using Gunicorn.

## 10. Create .dockerignore

Create `.dockerignore`:

```text
.venv
venv
__pycache__
*.pyc
.git
.env
```

This excludes unnecessary files from the Docker build context.

## 11. Build the Docker Image

Run the command from the project directory:

```bash
docker build -t breast-cancer-app .
```

Explanation:

- `docker build`: builds a Docker image.
- `-t breast-cancer-app`: assigns the image name.
- `.`: uses the current directory as the build context.

Check the image:

```bash
docker images
```

## 12. Run the Docker Container

Execute:

```bash
docker run -d --name breast-cancer-container -p 5000:5000 breast-cancer-app
```

Open the application in your browser:

http://localhost:5000

The port mapping `-p 5000:5000` connects host port 5000 to container port 5000.

## 13. Useful Docker Commands

Check running containers:

```bash
docker ps
```

View logs:

```bash
docker logs breast-cancer-container
```

Stop the container:

```bash
docker stop breast-cancer-container
```

Remove the stopped container:

```bash
docker rm breast-cancer-container
```

## 14. Troubleshooting

### Docker is not running

Start Docker Desktop and wait until the Docker engine is ready.

### Model file not found

Run the training script:

```bash
python train_model.py
```

### Missing Python module

Add the required package to `requirements.txt`, rebuild the image, and recreate the container.

### Port is already in use

Use another host port:

```bash
docker run -d --name breast-cancer-container -p 5001:5000 breast-cancer-app
```

Open http://localhost:5001.

### Container fails to start

Inspect the logs:

```bash
docker logs breast-cancer-container
```

Check the model path, dependencies, and Flask application name.

## 15. Security and Responsible Use

- Do not enable Flask debug mode in production.
- Validate all inputs.
- Do not load model files from untrusted sources.
- Avoid logging sensitive health information.
- Use authentication and HTTPS for public deployments.
- Do not use this educational application for medical diagnosis or treatment.

## 16. Conclusion

This project demonstrates the basic workflow for serving a machine learning model through Flask and packaging the application in Docker.

Future improvements may include a complete HTML interface, automated testing, GitHub Actions CI/CD, Docker Hub or Amazon ECR publishing, and cloud deployment.

## License

Add an appropriate license for your source code and verify the dataset's terms before redistributing data or model artifacts.