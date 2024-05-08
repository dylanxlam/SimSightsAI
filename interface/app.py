from data_handler import cleaning


# app.py (Flask application)

from flask import request
from data_handler import read_data

@app.route("/upload", methods=["POST"])
def handle_upload():
    # Access uploaded file
    uploaded_file = request.files["data_file"]  # Adjust key based on upload form

    # Read data using the function from data_handler.py
    try:
        data = read_data(uploaded_file)
        # Perform further data processing or visualization using the loaded data (pandas DataFrame)
    except ValueError as e:
        # Handle error if unsupported file format is uploaded
        return f"Error: {e}"

    return "Data uploaded successfully!"  # Or redirect to data exploration page
