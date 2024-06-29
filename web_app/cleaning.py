########################################################################################
# Import Statements
########################################################################################
import pandas as pd
import json
import numpy as np


########################################################################################
# Reading Data
########################################################################################

def read_data(filepath):
    """
    Reads data from a specified file path (supports CSV, Excel, TSV, JSON).

    Args:
        filepath (str): The path to the data file.

    Returns:
        pandas.DataFrame (or list/dict): The loaded data in a suitable format.
    """
    # Identify file format based on filename extension
    if filepath.endswith(".csv"):
        data = pd.read_csv(filepath)
    elif filepath.endswith(".xlsx"):
        data = pd.read_excel(filepath)
    elif filepath.endswith(".tsv"):
        data = pd.read_csv(filepath, sep="\t")  # Use tab separator for TSV
    elif filepath.endswith(".json"):
        with open(filepath, 'r') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                raise ValueError("Invalid JSON format. Please check your data.")
    else:
        raise ValueError("Unsupported file format. Please specify a CSV, Excel, TSV, or JSON file.")

    return data





########################################################################################
# Handling Null Values
########################################################################################

def handle_missing_values(data):
  """
  Analyzes and handles missing values in a DataFrame based on user choices.

  Args:
      data (pandas.DataFrame): The DataFrame containing the data.

  Returns:
      pandas.DataFrame: The DataFrame with missing values handled.
  """

  # Check for missing values
  missing_values = data.isnull().sum()

  if missing_values.sum() == 0:
    print("No missing values detected in the data!")
    return data

  # Inform user about missing values
  print("Missing values detected in the following columns:")
  print(missing_values[missing_values > 0])
  print("\nIt's crucial to address missing values before further analysis.")

  while True:
    method_choice = input("\nChoose a method to handle missing values (deletion/imputation/quit): ").lower()

    if method_choice == "deletion":
      print("\nDeletion removes rows with missing values. This is simple but can lose data.")
      confirmation = input("Proceed with deletion? (y/n): ").lower()
      if confirmation == "y":
        data = data.dropna()
        print("Rows with missing values have been deleted.")
        return data
      else:
        print("Deletion cancelled.")

    elif method_choice == "imputation":
      for column in data.columns[data.isnull().any()]:
        print(f"\nHandling missing values in column: {column}")
        if data[column].dtype in ['int64', 'float64']:
          while True:
            impute_choice = input("Choose an imputation method (mean/median/mode): ").lower()
            if impute_choice in ["mean", "median", "mode"]:
              if impute_choice == "mean":
                data[column].fillna(data[column].mean(), inplace=True)
              elif impute_choice == "median":
                data[column].fillna(data[column].median(), inplace=True)
              else:
                data[column].fillna(data[column].mode()[0], inplace=True)
              print(f"Missing values in {column} imputed using {impute_choice}.")
              break
            else:
              print("Invalid choice. Please choose mean, median, or mode.")
        else:
          while True:
            cat_method_choice = input("Choose a method for categorical data (mode/create_category): ").lower()
            if cat_method_choice == "mode":
              data[column].fillna(data[column].mode()[0], inplace=True)
              print(f"Missing values in {column} imputed using mode.")
              break
            elif cat_method_choice == "create_category":
              data[column].fillna("Unknown", inplace=True)
              print(f"Missing values in {column} replaced with 'Unknown' category.")
              break
            else:
              print("Invalid choice. Please choose mode or create_category.")

      print("\nAll missing values have been handled.")
      return data

    elif method_choice == "quit":
      print("Exiting without handling missing values.")
      return data

    else:
      print("Invalid method chosen. Please choose deletion, imputation, or quit.")



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
        outlier_count = ((numerical_cols[col] < lower_bound) & (numerical_cols[col] > upper_bound)).sum()


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
  else:
     print("No potential date formatting inconsistencies found.")

  # Check for currency formatting inconsistencies
  currency_cols = [col for col in data if pd.api.types.is_numeric_dtype(data[col]) and any(char in data[col] for char in r"$£€¥₱")]
  if currency_cols:
    print("Found potential currency formatting inconsistencies in columns:")
    for col in currency_cols:
      print(f"  - {col} (mixed currency symbols or no symbol)")
    print("  (Inconsistent currency formatting can hinder analysis.)\n")
  else:
    print("No potential currency formatting inconsistencies found")

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
