import requests
import json


url = "http://127.0.0.1:5001/invocations"

payload = {
    "dataframe_split": {
        "columns": ["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"],
        "data": [[7.4, 0.7, 0.0, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4]]
    }
}

response = requests.post(url, headers={"Content-Type": "application/json"}, data=json.dumps(payload))

print("Status code:", response.status_code)
print("Prediction:", response.json())
