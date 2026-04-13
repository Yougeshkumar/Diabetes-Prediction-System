from flask import Flask, render_template, request
import pandas as pd
import joblib
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load pre-trained models and scaler
clf = joblib.load('models/diabetes_classifier.pkl')  # Classification model
scaler = joblib.load('models/scaler.pkl')            # Scaler used during training

# Function to predict diabetes and risk level, and provide precautions
def predict_diabetes(data):
    # Create DataFrame from input data
    input_data = pd.DataFrame([data], columns=[
        'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
    ])
    
    # Scale the input data
    input_data_scaled = scaler.transform(input_data)

    # Predict diabetes outcome (0 or 1)
    outcome = clf.predict(input_data_scaled)[0]

    # If the person has diabetes (outcome == 1), return that outcome immediately
    if outcome == 1:
        precautions = "You have diabetes. Please follow a healthy diet, regular exercise, and consult with your doctor regularly."
        return outcome, precautions
    else:
        # If the person does not have diabetes, classify risk level
        glucose, bmi, age = data['Glucose'], data['BMI'], data['Age']

        # Determine risk level based on glucose, BMI, and age
        if glucose > 140 or bmi > 30 or age > 40:
            risk_level = 'High'
            precautions = '''
            **High Risk Precautions**:
            - Reduce sugar, refined carbs, and processed foods.
            - Exercise 30 minutes a day, 5 days a week.
            - Aim for 5-10% weight loss.
            - Regular monitoring of glucose levels.
            '''
        elif 110 <= glucose <= 140 or 25 <= bmi <= 30 or 35 <= age <= 40:
            risk_level = 'Moderate'
            precautions = '''
            **Moderate Risk Precautions**:
            - Switch to a low-glycemic diet.
            - Engage in physical activities like walking or swimming.
            - Gradual weight loss.
            - Regular check-ups and hydration.
            '''
        else:
            risk_level = 'Low'
            precautions = '''
            **Low Risk Precautions**:
            - Maintain a balanced diet with fruits and vegetables.
            - Regular physical activity.
            - Annual glucose testing and preventive screenings.
            '''
        return outcome, risk_level, precautions

# The home route
@app.route('/')
def index():
    return render_template('home.html')

# Prediction page route
@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if request.method == 'POST':
        # Retrieve form data and convert to float values
        data = {
            'Pregnancies': float(request.form['pregnancies']),
            'Glucose': float(request.form['glucose']),
            'BloodPressure': float(request.form['blood_pressure']),
            'SkinThickness': float(request.form['skin_thickness']),
            'Insulin': float(request.form['insulin']),
            'BMI': float(request.form['bmi']),
            'DiabetesPedigreeFunction': float(request.form['diabetes_pedigree_function']),
            'Age': float(request.form['age'])
        }

        # Predict diabetes status and risk
        outcome, risk_or_precautions, precautions = predict_diabetes(data)

        # Return result based on prediction outcome
        if outcome == 1:
            result = f"The person has diabetes. Precautions: {precautions}"
        else:
            result = f"The person does not have diabetes. Risk level: {risk_or_precautions}. {precautions}"

        return render_template('prediction.html', result=result)

    return render_template('prediction.html', result='')

if __name__ == '__main__':
    app.run(debug=True)
