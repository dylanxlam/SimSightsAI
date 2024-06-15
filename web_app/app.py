from flask import Flask, request, session

# Import functions from other files
from cleaning import read_data, handle_missing_values, identify_and_handle_outliers
from eda import visualize_numerical
from preprocessing import preprocess_data
from modeling import train_model
from analysis import generate_classification_report

app = Flask(__name__)
app.secret_key = "your_secret_key"  # Set a secret key for session security

@app.route("/upload", methods=["POST"])
def upload_data():
  # Get uploaded file
  file = input("Please upload your data file. We support CSV, Excel, TSV, and JSON")

  # Process the uploaded file (e.g., read content)
  data = read_data(file)

  # Store data in session (replace with actual storage logic)
  session["uploaded_data"] = data
  return "Data Uploaded Successfully!"

@app.route("/clean_data", methods=["POST"])
def clean_data_route():
  # Retrieve data from session (replace with actual retrieval logic)
  data = session.get("uploaded_data")
  # Call cleaning functions from cleaning.py
  cleaned_data = handle_missing_values(data)
  cleaned_data = identify_and_handle_outliers(cleaned_data)
  # ... call other cleaning functions
  # Update session with cleaned data (optional)
  session["cleaned_data"] = cleaned_data
  return "Data Cleaning Completed!"

# Clean the data
@app.route("/clean_data", methods=["POST"])
def clean_data_route():
  # Get data from user input (replace with actual data handling)
  data = request.form.get("data")
  cleaned_data = clean_data(data)

# Explore data
@app.route("/exploratory_data_analysis", methods=["POST"])
def clean_data_route():
  # Get data from user input (replace with actual data handling)
  data = request.form.get("data")
  visualize_numerical(data)

# Preprocess the data
@app.route("/preprocess_data", methods=["POST"])
def preprocess_data_route():
  # Get data from user input (replace with actual data handling)
  data = request.form.get("data")
  features, target = preprocess_data(cleaned_data)

@app.route("/train_model", methods=["POST"])
def train_pipeline():
  """
  Handles user request to train the model.
  """
  # Get data from user input (replace with actual data handling)
  data = request.form.get("data")

  # Preprocess the data
  features, target = preprocess_data(cleaned_data)

  # Train the model (assuming chosen_model is defined elsewhere)
  trained_model = train_model(features, target, chosen_model)

  # Model Analysis
  generate_classification_report(trained_model)

if __name__ == "__main__":
  app.run(debug=True)  # Set debug=False for production deployment
