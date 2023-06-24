from flask import Flask, request, jsonify
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
import json
app = Flask(__name__)

# Load the pre-trained AdaBoost model
with open('adaboost.pkl', 'rb') as file:
    adaboost = pickle.load(file)

# # Load the scaler used for feature scaling
# with open('/kaggle/working/scaler.pkl', 'rb') as file:
#     scaler = pickle.load(file)
scaler = StandardScaler()
@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the request
    input_data = request.get_json()
    print(input_data)
    # Preprocess the input data
    input_features = input_data['features']
    print(input_features)
    #input_features_scaled = scaler.fit_transform([input_features])
    print(input_features)
    # Make predictions using the pre-trained model
    prediction = adaboost.predict([input_features])
    print(prediction)
    # Create the response object
    response = jsonify({'prediction': str(prediction[0])})
    return response

if __name__ == '__main__':
    app.run()