########################################################################################
# Import Statements
########################################################################################
import pandas as pd
import numpy as np 
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import sympy as sp
import sklearn
from imblearn.under_sampling import RandomUnderSampler  # Import for undersampling
from imblearn.over_sampling import SMOTE  # Import for oversampling
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2  # Example statistical test


########################################################################################
# Data Type Conversion
########################################################################################


def convert_data_types(data):
  """
  Prints data types for each column and allows user to change them.

  Args:
      data (pandas.DataFrame): The DataFrame containing the data.

  Returns:
      pandas.DataFrame: The DataFrame potentially with changed data types.
  """

  data_types = data.dtypes

  # Explain data types in a dictionary for easy reference
  dtype_explanations = {
      'int64': "Integer (whole numbers, positive or negative)",
      'float64': "Decimal number",
      'object': "Text data (strings)",
      'category': "Categorical data (limited set of options)",
      'datetime64[ns]': "Date and time",
      'bool': "Boolean (True or False)"
  }

  # Print data types with explanations
  for col, dtype in data_types.items():
    print(f"- {col}: {dtype} ({dtype_explanations.get(dtype, 'Unknown')})")

  # Prompt user for data type changes
  change_dtypes = input("Would you like to change any data types (y/n)? ").lower()
  if change_dtypes == "y":
    while True:
      # Ask for column and desired data type
      col_to_change = input("Enter the column name to change the data type: ").lower()
      new_dtype = input("Enter the desired new data type (int64, float64, object, etc.): ").lower()

      # Check if column exists and new data type is valid
      if col_to_change in data.columns and new_dtype in dtype_explanations.keys():
        try:
          # Attempt conversion (handles potential errors)
          data[col_to_change] = data[col_to_change].astype(new_dtype)
          print(f"Data type for '{col_to_change}' changed to {new_dtype}.")
          # **Modified break logic:**
          break_loop = input("Do you want to convert another column (y/n)? ").lower()
          if break_loop != "y":
            break
        except (ValueError, TypeError) as e:
          print(f"Error converting '{col_to_change}' to {new_dtype}: {e}")
          # **Prompt to continue after error**
          continue_loop = input("Would you like to try converting another column (y/n)? ").lower()
          if continue_loop != "y":
            break
      else:
        print(f"Invalid column name or data type. Please try again.")

  return data


########################################################################################
# Normalizing/Scaling Data
########################################################################################
def scale(data):
  """
  Identifies skewed features, suggests corrections, and performs scaling/normalization.

  Args:
      data (pandas.DataFrame): The DataFrame containing the data.

  Returns:
      pandas.DataFrame: The transformed DataFrame with addressed skewness and scaling/normalization.
  """
  numerical_cols = data.select_dtypes(include=[np.number])
  skewed_cols = []  # List to store column names with skewness

  # Threshold for skewness (adjust as needed)
  skewness_threshold = 0.5

  for col in numerical_cols:
    # Calculate skewness
    skew = data[col].skew()
    if abs(skew) > skewness_threshold:
      skewed_cols.append(col)
      print(f"Column '{col}' appears skewed (skewness: {skew:.2f}).")


      # Inform decision-making
      print("Here's a brief explanation of the available correction methods:")
      print("  - Log transformation (log(x + 1)): This method is often effective for right-skewed data (where values are concentrated on the left side of the distribution).")
      print("    It compresses the larger values and stretches the smaller ones, aiming for a more symmetrical distribution.")
      print("  - Square root transformation (sqrt(x)): This method can be helpful for moderately skewed data, positive-valued features, or data with a large number of zeros.")
      print("    It reduces the influence of extreme values and can bring the distribution closer to normality.")
      print("**Please consider the characteristics of your skewed feature(s) when making your choice.**")
      print("If you're unsure, you can experiment with both methods and compare the results visually (e.g., using histograms) to see which one normalizes the data more effectively for your specific case.")

      # User prompt for addressing skewness
      action = input("Do you want to address the skewness (y/n)? ").lower()
      if action == "y":
        

        # User chooses to address skewness
        while True:  # Loop until a valid choice is made
          fix_method = input("Choose a correction method (log/sqrt/none): ").lower()
          if fix_method in ["log", "sqrt"]:
            # Apply transformation (log or sqrt)
            if fix_method == "log":
              data[col] = np.log(data[col] + 1)  # Avoid log(0) errors by adding 1
              print(f"Applied log transformation to column '{col}'.")
            else:
              data[col] = np.sqrt(data[col])
              print(f"Applied square root transformation to column '{col}'.")
            break  # Exit the loop if a valid choice is made
          else:
            print("Invalid choice. Please choose 'log', 'sqrt', or 'none'.")

      else:
        print(f"Skewness in '{col}' remains unaddressed.")
    
    if not skewed_cols:
      print("No significant skewness detected in numerical columns.")

  # User prompt for scaling/normalization (if applicable)
  if len(numerical_cols) > 0:

    print("Here's a brief explanation of the available scaling/normalization methods:")
    print("  - Standard scaling: This method transforms features by subtracting the mean and dividing by the standard deviation.")
    print("    This results in features centered around zero with a standard deviation of 1.")
    print("    It's suitable for algorithms that assume a normal distribution of features (e.g., Logistic Regression, Support Vector Machines).")
    print("  - Min-max scaling: This method scales each feature to a specific range, typically between 0 and 1.")
    print("    It achieves this by subtracting the minimum value and then dividing by the difference between the maximum and minimum values in the feature.")
    print("    This can be useful for algorithms that are sensitive to the scale of features (e.g., K-Nearest Neighbors).")
    print("**Choosing the right method depends on your data and the algorithm you're using.**")
    print("  - If you're unsure about the underlying distribution of your data, standard scaling might be a safer choice as it doesn't make assumptions about normality.")
    print("  - If your algorithm is sensitive to feature scales and doesn't assume normality, min-max scaling might be preferable.")
    print("Consider the characteristics of your data and algorithm when making your decision. You can also experiment with both methods")
    print("and compare the results using model performance metrics to see which one works best for your specific case.")

    action = input("Do you want to scale or normalize the numerical features (y/n)? ").lower()
    if action == "y":

      while True:  # Loop until a valid choice is made
        method = input("Choose scaling/normalization method (standard/minmax/skip): ").lower()
        if method in ["standard", "minmax"]:
          scaler = None  # Initialize scaler outside the loop (prevents recreation)
          if method == "standard":
            scaler = StandardScaler()
          else:
            scaler = MinMaxScaler(feature_range=(0, 1))

          # **Fix:** Transform data directly, avoiding indexing with numerical values
          transformed_data = scaler.fit_transform(data[numerical_cols])
          data[numerical_cols] = transformed_data

          print(f"Applied {method} scaling to numerical features.")
          break  # Exit the loop if a valid choice is made

        elif method == "skip":
          print("Skipping scaling/normalization.")
          break
        else:
          print("Invalid choice. Please choose 'standard', 'minmax', or 'skip'.")


  if not skewed_cols:
    print("No significant skewness detected in numerical columns.")
  return data


########################################################################################
# Creating Interaction Features
########################################################################################
def create_interaction_features(data, categorical_cols=None):
  """
  Creates interaction features from categorical columns in a DataFrame.

  Args:
      data (pandas.DataFrame): The DataFrame containing the data.
      categorical_cols (list, optional): A list of column names to consider for interaction features. If None, all categorical columns will be used. Defaults to None.

  Returns:
      pandas.DataFrame: The DataFrame with additional interaction features.
  """

  if categorical_cols is None:
    categorical_cols = [col for col in data.columns if data[col].dtype == 'category']

  if not categorical_cols:
    print("No categorical columns found in the data. Skipping interaction feature creation.")
    return data

  # Display recommendations before prompting user
  print("** Recommendations for Interaction Features:**")
  print("- Interaction features can capture complex relationships, potentially improving model performance.")
  print("- However, creating all possible interactions can lead to data sparsity and longer training times.")
  print("- Consider your domain knowledge to prioritize specific interactions.")
  print("- Start with a smaller set and use feature selection techniques for better interpretability.")


  # Get user confirmation to proceed
  action = input("Do you want to create interaction features from categorical columns (y/n)? ").lower()
  if action != "y":
    print("Skipping interaction feature creation.")
    return data

  # Prompt user to choose specific columns or create all possible interactions
  while True:
    choice = input("Choose interaction feature creation method (all/specific): ").lower()
    if choice in ["all", "specific"]:
      break
    else:
      print("Invalid choice. Please choose 'all' or 'specific'.")

  if choice == "all":
    # Create all pairwise interaction features
    for col1 in categorical_cols:
      for col2 in categorical_cols:
        if col1 != col2:
          data[f"{col1}_x_{col2}"] = data[col1].astype(str) + "_" + data[col2].astype(str)
    print("Created all possible pairwise interaction features.")

  else:
    # Prompt user to choose specific columns for interaction
    selected_cols = []
    while True:
      col_name = input("Enter a categorical column name (or 'done' to finish): ").lower()
      if col_name == "done":
        if not selected_cols:
          print("No columns selected. Skipping interaction feature creation.")
        else:
          for col1 in selected_cols:
            for col2 in selected_cols:
              if col1 != col2:
                data[f"{col1}_x_{col2}"] = data[col1].astype(str) + "_" + data[col2].astype(str)
          print(f"Created interaction features for selected columns: {', '.join(selected_cols)}")
        break
      elif col_name in categorical_cols:
        selected_cols.append(col_name)
        print(f"Column '{col_name}' added for interaction features.")
      else:
        print(f"Invalid column name: '{col_name}'. Please choose from categorical columns.")

  return data


########################################################################################
# Feature Binning
########################################################################################
def create_feature_bins(data, continuous_cols=None, n_bins=5):
  """
  Creates bins (intervals) for continuous features in a DataFrame.

  Args:
      data (pandas.DataFrame): The DataFrame containing the data.
      continuous_cols (list, optional): A list of column names to bin. If None, all continuous columns will be considered. Defaults to None.
      n_bins (int, optional): The number of bins to create for each feature. Defaults to 5.

  Returns:
      pandas.DataFrame: The DataFrame with new bin features (categorical).
  """

  if continuous_cols is None:
    continuous_cols = [col for col in data.columns if data[col].dtype in ['float64', 'int64']]

  if not continuous_cols:
    print("No continuous features found in the data. Skipping binning.")
    return data

  # Get user confirmation to proceed
  action = input("Do you want to create bins for continuous features (y/n)? ").lower()
  if action != "y":
    print("Skipping binning.")
    return data

  # Allow user to choose specific columns or bin all continuous features
  while True:
    choice = input("Choose binning method (all/specific): ").lower()
    if choice in ["all", "specific"]:
      break
    else:
      print("Invalid choice. Please choose 'all' or 'specific'.")

  if choice == "all":
    # Bin all continuous features
    for col in continuous_cols:
      bins = pd.cut(data[col], bins=n_bins, labels=False) + 1  # Add 1 for informative bin names
      data[f"binned_{col}"] = bins.astype("category")
      print(f"Created bins for feature '{col}'.")

  else:
    # Prompt user to choose specific columns for binning
    selected_cols = []
    while True:
      col_name = input("Enter a continuous feature name (or 'done' to finish): ").lower()
      if col_name == "done":
        if not selected_cols:
          print("No columns selected. Skipping binning.")
        else:
          for col in selected_cols:
            bins = pd.cut(data[col], bins=n_bins, labels=False) + 1  # Add 1 for informative bin names
            data[f"binned_{col}"] = bins.astype("category")
            print(f"Created bins for feature '{col}'.")
        break
      elif col_name in continuous_cols:
        selected_cols.append(col_name)
        print(f"Feature '{col_name}' added for binning.")
      else:
        print(f"Invalid column name: '{col_name}'. Please choose from continuous features.")

  return data


########################################################################################
# Feature Creation
########################################################################################
def create_custom_features(data):
  """
  Allows users to define and create custom features from existing features.

  Args:
      data (pandas.DataFrame): The DataFrame containing the data.

  Returns:
      pandas.DataFrame: The DataFrame with additional custom features.
  """

  print("** Feature Creation Options:")
  print("- Define a new feature using existing features with mathematical expressions.")
  print("- Create interaction features from categorical columns.")  # Reference existing function

  while True:
    choice = input("Choose a feature creation method (expression/interaction/none): ").lower()
    if choice in ["expression", "interaction", "none"]:
      break
    else:
      print("Invalid choice. Please choose 'expression', 'interaction', or 'none'.")

  if choice == "expression":
    # Feature creation using expressions
    while True:
      expression = input("Enter a mathematical expression using existing feature names (or 'done' to finish): ")
      if expression == "done":
        break

      # Validate expression using symbolic math library (optional)
      try:
        sp.sympify(expression)  # Raises an error for invalid expressions (optional)
      except (TypeError, NameError):
        print("Invalid expression. Please use existing feature names and basic mathematical operators (+, -, *, /).")
        continue

      # Create and add the new feature
      new_feature_name = input("Enter a name for the new feature: ")
      try:
        data[new_feature_name] = eval(expression)  # Evaluate the expression on the DataFrame
        print(f"Created new feature: '{new_feature_name}'")
        break  # Exit the loop if expression is valid
      except (NameError, SyntaxError):
        print("Error evaluating expression. Please check for typos or invalid syntax.")

  elif choice == "interaction":
    # Call the existing create_interaction_features function (assuming it's defined)
    data = create_interaction_features(data.copy())  # Avoid modifying original data

  else:
    print("Skipping custom feature creation.")

  return data


########################################################################################
# Encoding
########################################################################################
def create_one_hot_encoding(data, categorical_cols=None):
  """
  Creates one-hot encoded features from categorical columns in a DataFrame.

  Args:
      data (pandas.DataFrame): The DataFrame containing the data.
      categorical_cols (list, optional): A list of column names to encode. If None, all categorical columns will be considered. Defaults to None.

  Returns:
      pandas.DataFrame: The DataFrame with additional one-hot encoded features.
  """

  if categorical_cols is None:
    categorical_cols = [col for col in data.columns if data[col].dtype == "object"]

  if not categorical_cols:
    print("No categorical features found in the data. Skipping one-hot encoding.")
    return data

  print("One-hot encoding is a technique for representing categorical features (like 'color' or 'size') as separate binary features.")
  print("Imagine a feature 'color' with values 'red', 'green', and 'blue'. One-hot encoding would create three new features:")
  print("  - 'color_red' (1 if the color is red, 0 otherwise)")
  print("  - 'color_green' (1 if the color is green, 0 otherwise)")
  print("  - 'color_blue' (1 if the color is blue, 0 otherwise)")
  print("This allows machine learning models to understand the relationships between these categories more effectively.")
  print("However, one-hot encoding can increase the number of features in your data significantly, which might require more computational resources.")


  # Get user confirmation to proceed
  action = input("Do you want to create one-hot encoded features (y/n)? ").lower()
  if action != "y":
    print("Skipping one-hot encoding.")
    return data

  # Informative message about one-hot encoding
  print("One-hot encoding will create a separate binary feature for each unique category in a categorical column.")

  # Option to choose all or specific categorical features
  while True:
    choice = input("Choose encoding method (all/specific): ").lower()
    if choice in ["all", "specific"]:
      break
    else:
      print("Invalid choice. Please choose 'all' or 'specific'.")

  if choice == "all":
    # Encode all categorical features
    data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)
    print("Created one-hot encoded features for all categorical columns.")

  else:
    # Prompt user to choose specific columns for encoding
    selected_cols = []
    while True:
      col_name = input("Enter a categorical feature name (or 'done' to finish): ").lower()
      if col_name == "done":
        if not selected_cols:
          print("No columns selected. Skipping one-hot encoding.")
        else:
          data = pd.get_dummies(data, columns=selected_cols, drop_first=True)
          print(f"Created one-hot encoded features for selected columns.")
        break
      elif col_name in categorical_cols:
        selected_cols.append(col_name)
        print(f"Feature '{col_name}' added for one-hot encoding.")
      else:
        print(f"Invalid column name: '{col_name}'. Please choose from categorical features.")

  return data


def create_label_encoding(data, categorical_cols=None):
  """
  Creates label encoded features from categorical columns in a DataFrame.

  Args:
      data (pandas.DataFrame): The DataFrame containing the data.
      categorical_cols (list, optional): A list of column names to encode. If None, all categorical columns will be considered. Defaults to None.

  Returns:
      pandas.DataFrame: The DataFrame with label encoded features (integers).
  """

  if categorical_cols is None:
    categorical_cols = [col for col in data.columns if data[col].dtype == "object"]

  if not categorical_cols:
    print("No categorical features found in the data. Skipping label encoding.")
    return data

  print("Label encoding is a simpler way to handle categorical features. It assigns a unique number to each different category.")
  print("For example, a feature 'fruit' with values 'apple', 'banana', and 'orange' might be encoded as:")
  print("  - apple: 0")
  print("  - banana: 1")
  print("  - orange: 2")
  print("This allows machine learning models to process the data more easily. However, it's important to be aware of a potential drawback:")
  print("  - Label encoding might treat higher numbers as more 'important' even if the categories have no inherent order.")
  print("For example, 'orange' (encoded as 2) might seem 'better' than 'apple' (encoded as 0) to the model, even though they are just different fruits.")
  print("If the order of your categories doesn't matter, label encoding can be a good choice. But if the order is important, you might want to consider other encoding techniques.")


  # Get user confirmation to proceed
  action = input("Do you want to create label encoded features (y/n)? ").lower()
  if action != "y":
    print("Skipping label encoding.")
    return data

  # Informative message about label encoding
  print("Label encoding assigns a unique integer value to each category in a categorical column.")
  print("** Caution:** This might introduce unintended ordering between categories.")

  # Option to choose all or specific categorical features
  while True:
    choice = input("Choose encoding method (all/specific): ").lower()
    if choice in ["all", "specific"]:
      break
    else:
      print("Invalid choice. Please choose 'all' or 'specific'.")

  if choice == "all":
    # Encode all categorical features
    for col in categorical_cols:
      le = sklearn.preprocessing.LabelEncoder()
      data[col] = le.fit_transform(data[col])
    print("Created label encoded features for all categorical columns.")

  else:
    # Prompt user to choose specific columns for encoding
    selected_cols = []
    while True:
      col_name = input("Enter a categorical feature name (or 'done' to finish): ").lower()
      if col_name == "done":
        if not selected_cols:
          print("No columns selected. Skipping label encoding.")
        else:
          for col in selected_cols:
            le = sklearn.preprocessing.LabelEncoder()
            data[col] = le.fit_transform(data[col])
          print(f"Created label encoded features for selected columns.")
        break
      elif col_name in categorical_cols:
        selected_cols.append(col_name)
        print(f"Feature '{col_name}' added for label encoding.")
      else:
        print(f"Invalid column name: '{col_name}'. Please choose from categorical features.")

  return data





########################################################################################
# Handling Class Imbalance
########################################################################################
def handle_class_imbalance(data, target_column):
  """
  Provides options to handle class imbalance in a dataset.

  Args:
      data (pandas.DataFrame): The DataFrame containing the data.
      target_col (str): The name of the column containing the target variable.

  Returns:
      pandas.DataFrame: The DataFrame with potentially balanced classes.
  """

  # Display class distribution
  print("** Class Distribution:")
  class_counts = data[target_column].value_counts().sort_values(ascending=False)
  print(class_counts)

  # Check for imbalance
  majority_class = class_counts.index[0]
  majority_count = class_counts.iloc[0]
  imbalanced = majority_count / len(data) > 0.5  # Ratio check for imbalance

  if not imbalanced:
    print("Class distribution seems balanced. Skipping imbalance handling.")
    return data

  # Explain class imbalance
  print("\n** What is Class Imbalance?**")
  print("In machine learning, class imbalance occurs when a classification task has a significant skew")
  print("in the number of examples between different classes. Typically, one class (the majority class)")
  print("has many more examples than the other classes (the minority class).")
  print("This imbalance can lead to models that are biased towards the majority class and perform poorly")
  print("on the minority class.")

  # Get user choice for handling imbalance
  print("** Handling Class Imbalance:")
  print("- Undersampling (reduce majority class size)")
  print("  - Recommended if the majority class might be noisy or irrelevant.")
  print("- Oversampling (increase minority class size)")
  print("  - Recommended if the minority class is informative and you have enough data.")
  print("  - We will use the Synthetic Minority Oversampling Technique (SMOTE) for oversampling to avoid overfitting.")
  print("- No action (continue with imbalanced data)")
  print("  - Only recommended if the class imbalance doesn't significantly affect the model.")


  while True:
    choice = input("Choose an option (undersample/oversample/none): ").lower()
    if choice in ["undersample", "oversample", "none"]:
      break
    else:
      print("Invalid choice. Please choose 'undersample', 'oversample', or 'none'.")

  if choice == "none":
    print("Continuing with imbalanced data.")
    return data

  # Handle undersampling or oversampling based on user choice
  if choice in ["undersample", "oversample"]:
    print(f"Selected '{choice}'.")
    sampling_ratio = float(input("Enter desired sampling ratio (between 0 and 1): "))
    if sampling_ratio <= 0 or sampling_ratio > 1:
      print("Invalid sampling ratio. Please enter a value between 0 and 1.")
      return data  # Avoid errors with invalid ratio

    if choice == "undersample":
      rus = RandomUnderSampler(sampling_strategy={majority_class: int(sampling_ratio * majority_count)})
      data = rus.fit_resample(data, data[target_column])
      print(f"Undersampled majority class to {int(sampling_ratio * majority_count)} samples.")
    else:
      sm = SMOTE(sampling_strategy={target_column: "auto"})
      data = sm.fit_resample(data, data[target_column])
      print(f"Oversampled minority class to match the majority class size.")

  # Display final class distribution
  print("** Final Class Distribution:")
  class_counts = data[target_column].value_counts().sort_values(ascending=False)
  print(class_counts)

  return data


########################################################################################
# Feature selection
########################################################################################

def feature_selection(data, target_column):
  """
  Provides options for feature selection in machine learning tasks.

  Args:
      data (pandas.DataFrame): The DataFrame containing the data.
      target_col (str): The name of the column containing the target variable.

  Returns:
      pandas.DataFrame: The DataFrame with potentially reduced features.
  """

  # Display initial information
  print("** Feature Selection helps identify the most relevant features for your machine learning model.")
  print("It can improve model performance, reduce training time, and make the model easier to interpret.")

  # Get user preference for selection method
  print("\n** Feature Selection Methods:")
  print("- Filter Methods (based on statistical tests for individual features)")
  print("- Wrapper Methods (use a machine learning model to evaluate feature subsets)")
  print("- Embedded Methods (integrated within a machine learning model)")
  print("\n** We will focus on Filter Methods for this session.**")

  while True:
    choice = input("Do you want to proceed with Filter Methods (y/n)? ").lower()
    if choice in ["y", "n"]:
      break
    else:
      print("Invalid choice. Please choose 'y' or 'n'.")

  if choice == "n":
    print("Skipping feature selection. Using all features.")
    return data

  # Filter Method Selection
  print("\n** Filter Methods Options:")
  print("- Select K Best (choose a specific number of features)")
  print("- Select Percentile (choose a percentage of features)")
  print("** We will use Select K Best for this session.**")

  # Select K Best configuration
  while True:
    try:
      k = int(input("Enter the desired number of features to select (integer): "))
      if k > 0:
        break
      else:
        print("Invalid number. Please enter a positive integer.")
    except ValueError:
      print("Invalid input. Please enter an integer.")

  X = data.drop(target_column, axis=1)  # Separate features (X) and target (y)
  y = data[target_column]

  selector = SelectKBest(chi2, k=k)  # Use chi-square test for filter
  selector.fit(X, y)
  selected_features = X.columns[selector.get_support(indices=True)]

  # Informative output
  print(f"\n** Selected Features using SelectKBest (chi-square):**")
  for feature in selected_features:
    print(f"- {feature}")

  # User confirmation for using selected features
  print("\n** These features will be used for model training.**")
  while True:
    choice = input("Continue with selected features (y/n)? ").lower()
    if choice in ["y", "n"]:
      break
    else:
      print("Invalid choice. Please choose 'y' or 'n'.")

  if choice == "n":
    print("Original features will be used for model training.")
    return data

  # Return DataFrame with selected features
  return data[selected_features]


