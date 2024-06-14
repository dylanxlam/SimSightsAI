########################################################################################
# Import Statements
########################################################################################
import pandas as pd
import json
import numpy as np


########################################################################################
# Reading Data
########################################################################################
file = input("Please upload your data file. We support CSV, Excel, TSV, and JSON")

def read_data(file):
    """
    Reads data from uploaded file (supports CSV, Excel, TSV, JSON).

    Args:
        file (object): The uploaded file object from Flask request.

    Returns:
        pandas.DataFrame (or list/dict): The loaded data in a suitable format.
    """
    # Identify file format based on filename extension or MIME type (consider using magic library)
    if file.filename.endswith(".csv"):
        data = pd.read_csv(file)
    elif file.filename.endswith(".xlsx"):
        data = pd.read_excel(file)
    elif file.filename.endswith(".tsv"):
        data = pd.read_csv(file, sep="\t")  # Use tab separator for TSV
    elif file.filename.endswith(".json"):
        try:
            data = json.load(file)  # Assuming JSON data represents a list or dictionary
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON format. Please check your data.")
    else:
        raise ValueError("Unsupported file format. Please upload CSV, Excel, TSV, or JSON files.")

    return data


########################################################################################
# Handling Null Values
########################################################################################

# Null Values for each column
missing_values = data.isnull().sum()
print("Missing/Null Values for Each Column/Feature of Your Data:\n", missing_values)

def handle_missing_values(data):
  """
  Analyzes and guides the user on handling missing values in a DataFrame.

  Args:
      data (pandas.DataFrame): The DataFrame containing the data.

  Returns:
      pandas.DataFrame: The DataFrame potentially with imputed missing values
                          based on the user's choice.
  """
  # Check for missing values

  # No missing values
  
  if missing_values.sum() == 0:
    print("Good, no null values to deal with in this dataset!")
  else:
    # Inform user about missing values and their importance
    print("There are missing values in your data! It's crucial to address them")
    print("before further analysis. Missing values can skew results and lead to")
    print("inaccurate conclusions. Let's handle them!")

    # Prompt user for null value technique (loop for repeated input validation)
    while True:
      method_choice = input("Choose a method to handle missing values (deletion, imputation, encoding): ").lower()

      # User chooses deletion
      if method_choice == "deletion":
        print("Deletion removes rows/columns with missing values. This is simple")
        print("but can lose data, especially if missingness is high. Are you sure?")
        confirmation = input("Proceed with deletion (y/n)? ").lower()
        if confirmation == "y":
          data = data.dropna()  # Drops rows with any missing values
          print("Missing values deleted!")
          break  # Exit the loop after successful deletion
        else:
          print("Deletion skipped based on your confirmation.")

      # User chooses imputation
      elif method_choice == "imputation":
        print("Imputation estimates missing values based on other data points.")
        print("There are different imputation techniques, each with advantages and disadvantages:")
        print("  - Mean/Median/Mode Imputation (simple but might not be suitable for skewed data).")
        print("  - Interpolation (estimates missing values based on surrounding values).")
        print("  - Model-based Imputation (uses machine learning to predict missing values,")
        print("     more complex but potentially more accurate).")
        impute_choice = input("Choose an imputation method (mean/median/mode/interpolation): ").lower()
        if impute_choice in ["mean", "median", "mode"]:
          # Simple imputation using mean/median/mode
          if impute_choice == "mean":
            imputation_strategy = np.mean
          elif impute_choice == "median":
            imputation_strategy = np.median
          else:
            imputation_strategy = np.mode
          data = data.fillna(method=imputation_strategy)  # Fill missing values with chosen strategy
          print(f"Missing values imputed using {impute_choice} strategy!")
          break  # Exit the loop after successful imputation
        elif impute_choice == "interpolation":
          # Interpolation (using linear interpolation here)
          data = data.interpolate("linear")  # Linear interpolation for missing values
          print("Missing values imputed using linear interpolation!")
          break  # Exit the loop after successful interpolation
        else:
          print("Invalid imputation method chosen. Skipping imputation.")

      # User chooses encoding
      elif method_choice == "encoding":
        print("Encoding creates a new feature indicating the presence or absence of a missing value.")
        print("This can be informative for some models but increases the number of features.")
        print("Are you sure you want to proceed with encoding?")
        confirmation = input("Proceed with encoding (y/n)? ").lower()
        if confirmation == "y":
          data = pd.get_dummies(data, dummy_na=True)  # Encode missing values as features
          print("Missing values encoded as new features!")
          break  # Exit the loop after successful encoding
        else:
          print("Encoding skipped based on your confirmation.")

      # Invalid method chosen (loop continues)
      else:
        print("Invalid method chosen. Please try again from the available options:")
        print("- deletion")
        print("- imputation")
        print("- encoding")

  return data


########################################################################################
# Handling Outliers
########################################################################################
def identify_and_handle_outliers(data):
    """
    Identifies outliers and prompts user for imputation, removal, or keeping outliers.

    Args:
        data (pandas.DataFrame): The DataFrame containing the data.

    Returns:
        pandas.DataFrame: The modified DataFrame (potentially with outliers left unchanged).
    """
    numerical_cols = data.select_dtypes(include=[np.number])
    outliers_exist = False  # Flag to track presence of outliers

    for col in numerical_cols:
        # Calculate quartiles and IQR
        Q1 = numerical_cols[col].quantile(0.25)
        Q3 = numerical_cols[col].quantile(0.75)
        IQR = Q3 - Q1

        # Identify outliers based on IQR outlier rule
        lower_bound = Q1 - (1.5 * IQR)
        upper_bound = Q3 + (1.5 * IQR)
        outlier_count = (numerical_cols[col] < lower_bound | numerical_cols[col] > upper_bound).sum()

    if outlier_count > 0:
        outliers_exist = True
        print(f"Found {outlier_count} potential outliers in column '{col}'.")
        print("""
        Outlier Treatment Options:

        * Imputation: Replaces outliers with estimates (mean, median, mode) to preserve data.
        * Removal: Removes rows containing outliers, suitable for errors or irrelevant data.
        * Keep: Leave outliers unchanged for further analysis (consider impact on results).

        Choosing the right option depends on the number of outliers, their impact on analysis, and data quality.
        """)
        action = input("Do you want to (i)mpute, (r)emove, or (k)eep outliers (i/r/k)? ").lower()
        if action == "i":
            # FUTURE DEVELOPMENT: See markdown below this cell to determine which imputation method to choose.
            # Choose imputation method
            print("""
            Choosing the Right Imputation Method:

            * **Mean:** Use mean if the data is normally distributed (consider histograms or normality tests). Mean is sensitive to outliers, so consider if there are extreme values that might distort the average.

            * **Median:** Use median if the data is skewed (uneven distribution) or has extreme outliers. Median is less sensitive to outliers compared to mean and represents the 'middle' value in the data.

            * **Mode:** Use mode for categorical data with a dominant value. Mode represents the most frequent value in the data and is suitable for non-numerical categories.
            """)
            imputation_method = input("Choose imputation method (mean/median/mode): ").lower()
            if imputation_method == "mean":
                data.loc[numerical_cols[col].index[numerical_cols[col] < lower_bound | numerical_cols[col] > upper_bound], col] = numerical_cols[col].mean()
                print(f"Imputing outliers in '{col}' with mean.")
            elif imputation_method == "median":
                data.loc[numerical_cols[col].index[numerical_cols[col] < lower_bound | numerical_cols[col] > upper_bound], col] = numerical_cols[col].median()
                print(f"Imputing outliers in '{col}' with median.")
            else:
                # Mode imputation (consider using libraries like scikit-learn for categorical data handling)
                data.loc[numerical_cols[col].index[numerical_cols[col] < lower_bound | numerical_cols[col] > upper_bound], col] = numerical_cols[col].mode()[0]  # Assuming single most frequent value
                print(f"Imputing outliers in '{col}' with mode (considering first most frequent value).")
        elif action == "r":
            # Remove rows with outliers
            data = data[~(numerical_cols[col] < lower_bound | numerical_cols[col] > upper_bound)]
            print(f"Removing rows with outliers in column '{col}'.")
        elif action == "k":
            print(f"Keeping outliers in column '{col}' for further analysis.")
        else:
            print(f"Invalid choice. Outliers in '{col}' remain unaddressed.")

    if not outliers_exist:
        print("No outliers detected in numerical columns.")

    return data


########################################################################################
# Handling Duplicates
########################################################################################
def handle_duplicates(data):
  """
  Identifies duplicates, explains options, and allows user choice for handling them.

  Args:
      data (pandas.DataFrame): The DataFrame containing the data.

  Returns:
      pandas.DataFrame: The DataFrame potentially with duplicates removed.
  """

  # Find duplicates
  duplicates = data.duplicated()

  # Check if any duplicates exist
  if not duplicates.any():
    print("No duplicate rows found in your data. Moving on...")
    return data

  # Print a sample of duplicates (avoid overwhelming the user)
  print("Found potential duplicate rows. Here are 5 samples:")
  print(data[duplicates].head())

  # Explain duplicate handling options
  print("\nHow would you like to handle these duplicates?")
  print("  1. Remove all duplicates (keeps the first occurrence)")
  print("  2. Keep all duplicates (may skew analysis)")
  print("  3. View all duplicates (for manual selection)")

  while True:
    choice = input("Enter your choice (1, 2, or 3): ")

    # Handle user choice
    if choice == "1":
      print("Removing all duplicates (keeping the first occurrence).")
      data = data.drop_duplicates()
      break  # Exit the loop after a valid choice
    elif choice == "2":
      print("Keeping all duplicates (may skew analysis).")
      break  # Exit the loop after a valid choice
    elif choice == "3":
      print("Here are all duplicates. Review and choose rows to keep (comma-separated indices):")
      print(data[duplicates])
      keep_indices = input("Enter indices of rows to KEEP (or 'all' to keep all): ")
      if keep_indices.lower() == "all":
        data = data[duplicates]  # Keep all duplicates
      else:
        try:
          # Convert user input to a list of integers (indices)
          keep_indices = [int(i) for i in keep_indices.split(",")]
          data = data.iloc[keep_indices]  # Keep rows based on indices
          print(f"Keeping rows with indices: {keep_indices}")
        except ValueError:
          print("Invalid input. Please enter comma-separated integers or 'all'.")
      break  # Exit the loop after a valid choice
    else:
      print("Invalid choice. Please enter 1, 2, or 3.")

  return data


########################################################################################
# Handling Inconsistent Formatting
########################################################################################
def handle_formatting(data):
  """
  Identifies inconsistent formatting and allows user choice for handling it.

  Args:
      data (pandas.DataFrame): The DataFrame containing the data.

  Returns:
      pandas.DataFrame: The DataFrame potentially with formatting inconsistencies fixed.
  """

  # Check for date formatting inconsistencies
  date_cols = [col for col in data if pd.api.types.is_datetime64_dtype(data[col])]
  if date_cols:
    print("Found potential date formatting inconsistencies in columns:")
    for col in date_cols:
      print(f"  - {col}")
    print("  (Inconsistent date formats can lead to errors during analysis.)\n")

  # Check for currency formatting inconsistencies
  currency_cols = [col for col in data if pd.api.types.is_numeric_dtype(data[col]) and any(char in data[col] for char in r"$£€¥₱")]
  if currency_cols:
    print("Found potential currency formatting inconsistencies in columns:")
    for col in currency_cols:
      print(f"  - {col} (mixed currency symbols or no symbol)")
    print("  (Inconsistent currency formatting can hinder analysis.)\n")

  # Offer choices if inconsistencies found
  if date_cols or currency_cols:
    choice = input("Would you like to attempt fixing these formatting issues (y/n)? ")
    if choice.lower() == "y":
      # Fix formatting based on user choice
      for col in date_cols:
        valid_choice = False
        while not valid_choice:
          print(f"\nChoose a desired date format for '{col}':")
          print("  1. YYYY-MM-DD (e.g., 2024-05-26)")
          print("  2. MM-DD-YYYY (e.g., 05-26-2024)")
          print("  3. D/M/YYYY (e.g., 26/05/2024)")
          print("  4. YYYY/MM/DD (e.g., 2024/05/26)")
          format_choice = input("Enter your choice (1, 2, etc.): ")
          try:
            # Convert to chosen date format (assuming choices 1, 2, 4, and 5)
            if format_choice in ("1", "2", "3", "4"):
              if format_choice == "1":
                data[col] = pd.to_datetime(data[col], format="%Y-%m-%d")
              elif format_choice == "2":
                data[col] = pd.to_datetime(data[col], format="%m-%d-%Y")
              elif format_choice == "3":
                data[col] = pd.to_datetime(data[col], format="%d/%m/%Y")
              elif format_choice == "4":
                data[col] = pd.to_datetime(data[col], format="%Y/%m/%d")
              valid_choice = True
            else:
              print("Invalid choice. Please choose from options 1-5.")
          except ValueError:
            print(f"Error parsing dates in '{col}'. Keeping existing format.")

      for col in currency_cols:
        print(f"\nChoose a desired currency symbol for '{col}':")
        print("  1. USD ($)")
        print("  2. EUR (€)")
        print("  3. No symbol")
        symbol_choice = input("Enter your choice (1, 2, or 3): ")
        if symbol_choice == "1":
          # Replace existing symbols with '$' (assuming text data)
          data[col] = data[col].str.replace(r"[£€¥₱]", "$", regex=True)
        elif symbol_choice == "2":
          data[col] = data[col].str.replace(r"[£¥₱$]", "€", regex=True)
        elif symbol_choice == "3":
          # Remove all currency symbols (assuming text data)
          data[col] = data[col].str.replace(r"[£€¥₱$]", "", regex=True)
        else:
          print("Invalid choice. Keeping existing formatting for", col)
      print("Formatting potentially fixed in some columns.")
    else:
      print("Keeping existing formatting (may cause issues during analysis).")

  return data
