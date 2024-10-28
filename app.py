from flask import Flask, render_template, request, jsonify
import plotly
import plotly.graph_objs as go
import numpy as np
import pandas as pd
import json
import os
import h5py
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Check the current working directory
print("Current Working Directory:", os.getcwd())

# Define the path for the model file
model_file = 'tmdb_model.h5'

# Verify if the model file is accessible
try:
    with h5py.File(model_file, 'r') as f:
        print("Model file is valid and accessible")
except OSError as e:
    print(f"Error accessing model file: {e}")
    exit(1)  # Exit if the model file cannot be opened

# Load the saved model
model = load_model(model_file)

# Load and preprocess the data
df = pd.read_csv('tmdb_movies_2023.csv')
df = df.dropna()  # Drop any rows with missing values
X = df[['budget', 'popularity', 'runtime']]

# Function to create predictions
def create_predictions(input_data):
    input_data = np.array(input_data).reshape(1, -1)  # Ensure input is shaped correctly
    prediction = model.predict(input_data)
    return prediction[0][0]

# Route for the dashboard
@app.route('/')
def dashboard():
    # Example plot: Revenue vs. Budget
    data = [
        go.Scatter(
            x=df['budget'],
            y=df['revenue'],
            mode='markers',
            marker=dict(size=10, color='rgba(152, 0, 0, .8)', line=dict(width=2, color='rgba(0, 0, 0, .8)'))
        )
    ]
    
    layout = go.Layout(title='Revenue vs Budget',
                       xaxis=dict(title='Budget'),
                       yaxis=dict(title='Revenue'))

    graphJSON = json.dumps({'data': data, 'layout': layout}, cls=plotly.utils.PlotlyJSONEncoder)
    
    return render_template('dashboard.html', graphJSON=graphJSON)

# Route for prediction form
@app.route('/predict', methods=['POST'])
def predict():
    budget = float(request.form['budget'])
    popularity = float(request.form['popularity'])
    runtime = float(request.form['runtime'])
    
    prediction = create_predictions([budget, popularity, runtime])
    
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)
