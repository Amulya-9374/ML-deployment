from flask import Flask, request, render_template, redirect, url_for
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import io

app = Flask(__name__)

from sklearn.preprocessing import LabelEncoder

def perform_analysis(data):
    # Check for expected columns and preprocess if necessary
    expected_columns = ['sepal.length', 'sepal.width', 'petal.length', 'petal.width', 'variety']
    if not all(col in data.columns for col in expected_columns):
        raise KeyError(f"Expected columns are missing. Available columns: {data.columns}")

    # Encode the categorical column 'species'
    if 'variety' in data.columns:
        label_encoder = LabelEncoder()
        data['variety'] = label_encoder.fit_transform(data['variety'])
    
    # Define features and target
    X = data[['sepal.length', 'sepal.width', 'petal.length', 'petal.width']]
    y = data['variety']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    
    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Calculate MSE
    mse = mean_squared_error(y_test, y_pred)
    return mse

@app.route('/app1', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        
        if file:
            data = pd.read_csv(io.StringIO(file.stream.read().decode('UTF8')), sep=',')
            mse = perform_analysis(data)
            return render_template('index.html', mse=mse)
    
    return render_template('index.html', mse=None)

if __name__ == "__main__":
    app.run(debug=True)
