from flask import Flask, render_template, request
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer

# Load the data from the Excel file
file_path = "D://final_data_diseases.xlsx"
df = pd.read_excel(file_path)

# Separate input features (symptoms) and target variable (diseases)
X = df['Symptoms']
y = df['Disease']

# Vectorize the symptoms using one-hot encoding
vectorizer = CountVectorizer()
X_encoded = vectorizer.fit_transform(X)

# Create a Random Forest classifier
rf_model = RandomForestClassifier()
rf_model.fit(X_encoded, y)

# Initialize the Flask application
app = Flask(__name__)


# Define the route for the home page
@app.route('/')
def home():
    return render_template('index.html')


# Define the route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get the symptoms entered by the user
    symptoms = request.form['symptoms']

    # Preprocess the symptoms
    symptoms_vectorized = vectorizer.transform([symptoms])

    # Make prediction using the trained model
    predicted_disease = rf_model.predict(symptoms_vectorized)

    # Render the prediction result
    return render_template('result.html', symptoms=symptoms, predicted_disease=predicted_disease)


# Run the Flask application
if __name__ == '__main__':
    app.run()
