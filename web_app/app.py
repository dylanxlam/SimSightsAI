from flask import Flask, request, session
import json
# Import functions from other files
from modeling import train_model
from analysis import generate_classification_report

app = Flask(__name__)
app.secret_key = "your_secret_key"  # Set a secret key for session security

@app.route("/upload_data", methods=["POST"])
def upload_data():
  # Get uploaded file
  uploaded_file = request.files.get("data_file")

  # Check if a file was uploaded
  if not uploaded_file:
    return "No file uploaded!", 400  # Return error if no file found

  # Import read_data function
  from cleaning import read_data

  # Get file extension for processing (replace with actual logic)
  file_extension = uploaded_file.filename.split(".")[-1].lower()

  # Call read_data function to process the file
  try:
    data = read_data(uploaded_file)
  except (ValueError, json.JSONDecodeError) as e:
    return f"Error processing file: {e}", 400  # Return error with details

  # Store data in session (consider alternative storage)
  session["uploaded_data"] = data.to_dict()  

  return "Data Uploaded Successfully!"



@app.route("/clean_data", methods=["GET", "POST"])
def clean_data_route():
  if request.method == "GET":
    # Display the missing value handling form
    return render_template("handle_missing_values.html")
  else:
    # Access data from session (replace with your data loading logic)
    data = session.get("uploaded_data")

    # Check for data existence (replace with your error handling)
    if not data:
      return "Error: No data uploaded yet!"

  # Check if data exists in session (already mentioned in previous response)
  if not data:
    return "No data uploaded yet. Please upload your data first.", 400

  # Import cleaning functions from cleaning.py
  from cleaning import handle_missing_values, identify_and_handle_outliers, handle_duplicates, handle_formatting

  # Access user choice from form
  method_choice = request.form.get("method")

  # Call cleaning function with user choice
  cleaned_data = handle_missing_values(data.copy(), method_choice)

  # Call cleaning functions
  identify_and_handle_outliers(cleaned_data)
  handle_duplicates(cleaned_data)
  handle_formatting(cleaned_data)

  # Update session with cleaned data (optional)
  session["cleaned_data"] = cleaned_data.to_dict()  # Assuming conversion to dictionary

  return "Data Cleaning Completed!"


@app.route("/exploratory_data_analysis", methods=["POST"])
def eda_route():
  # Retrieve cleaned data 
  cleaned_data = session.get("cleaned_data")  # Assuming session storage
  # (or) cleaned_data = request.form.get("cleaned_data")  # If using form

  # Check if data exists
  if not cleaned_data:
    return "No cleaned data found. Please upload and clean your data first.", 400

  # Data Descriptive Statistics
  data_descriptive_statistics = cleaned_data.describe()
  print("Descriptive Statistics of Your Data:\n", data_descriptive_statistics)

  # Import eda functions
  from eda import visualize_numerical, visualize_categorical, analyze_correlations

  # Call eda functions
  visualize_numerical(cleaned_data)
  visualize_categorical(cleaned_data)
  analyze_correlations(cleaned_data)

  # Consider user interaction for feature selection (optional)
  # if using a templating engine like Jinja2:
  #   return render_template("eda.html", data=cleaned_data)  # Pass data for template
  # (or) we could return a JSON response with data and options for feature selection

  return "Exploratory Data Analysis Completed!"



# Preprocess the data
@app.route("/preprocess_data", methods=["POST"])
def preprocess_data_route():
  # Retrieve cleaned data 
  cleaned_data = session.get("cleaned_data")  # Assuming session storage
  # (or) cleaned_data = request.form.get("cleaned_data")  # If using form

  # Check if data exists
  if not cleaned_data:
    return "No data found. Please upload and clean your data first.", 400

  # Data Types for Information
  data_types = cleaned_data.dtypes
  print("Data Types for Each Column in Your Data:\n", data_types)

  # Import preprocessing functions
  from preprocessing import convert_data_types, scale, create_interaction_feature, create_feature_bins, create_custom_features, create_one_hot_encoding, create_label_encoding, handle_class_imbalance, feature_selection

  # Preprocess the data
  preprocessed_data = convert_data_types(cleaned_data)
  scale(preprocessed_data)

  # Interaction Features (consider offering options or removing prompt)
  create_interaction_feature(preprocessed_data, categorical_cols=None)  # Consider offering options or removing

  # Feature Binning (consider offering options or removing prompt)
  create_feature_bins(preprocessed_data, continuous_cols=None, n_bins=5)  # Consider offering options or removing

  # Custom Feature Creation (consider implementing or removing)
  create_custom_features(preprocessed_data)  

  # Categorical Encoding
  create_one_hot_encoding(preprocessed_data, categorical_cols=None)  # Consider offering options or removing

  # Label Encoding
  create_label_encoding(preprocessed_data, categorical_cols=None)

  # Target Variable and Class Imbalance
  target_column = request.form.get("prompt", "Enter the name of the column containing the target variable (the variable you wish to predict/classify):")  # Use request.form.get with a default prompt
  if not target_column:
    return "Please specify the target variable column name.", 400
  handle_class_imbalance(preprocessed_data, target_column)

  # Feature Selection (consider using automated methods or offering options)
  feature_selection(preprocessed_data, target_column)

  # Store preprocessed data (replace with actual storage logic)
  session["preprocessed_data"] = preprocessed_data.to_dict() 

  return "Data Preprocessing Completed!"



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
