########################################################################################
# Import Statements
########################################################################################
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder



from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

def data_splitting(data, target_column, test_size=0.2, random_state=42):
  """
  Guides the user through splitting data into training, validation, and test sets 
  for machine learning tasks, allowing adjustment of training size.

  Args:
      data (pandas.DataFrame): The DataFrame containing the data.
      target_column (str): The name of the column containing the target variable.
      test_size (float, optional): The desired size for the test set (between 0 and 1, default: 0.2).
      random_state (int, optional): Sets the random seed for reproducibility (default: 42).

  Returns:
      tuple: A tuple containing the training, validation, and test DataFrames.
  """

  print("** Data Splitting is crucial for training and evaluating machine learning models.**")
  print("It separates your data into three sets: training, validation, and test.")
  print("The training set is used to build the model, the validation set is used to assess its performance on unseen data during training (hyperparameter tuning), and the test set provides a final evaluation on completely unseen data after training is complete.")

  print(f"\n** By default, we will use {test_size*100:.0f}% of your data for testing, and the remaining data will be split between training and validation sets using scikit-learn's train_test_split function.**")
  print(f"** Would you like to adjust the default test set size (currently {test_size:.2f})?**")
  while True:
    choice = input("Enter 'y' or 'n': ").lower()
    if choice in ["y", "n"]:
      break
    else:
      print("Invalid choice. Please choose 'y' or 'n'.")
  if choice == "y":
    while True:
      try:
        test_size = float(input("Enter the desired size for the test set (between 0 and 1): "))
        if 0 < test_size < 1:
          break
        else:
          print("Invalid input. Please enter a value between 0 and 1.")
      except ValueError:
        print("Invalid input. Please enter a number.")

  remaining_data = 1 - test_size
  print(f"\n** After allocating {test_size*100:.0f}% for testing, you have {remaining_data*100:.0f}% of data remaining for training and validation.**")
  print("** Would you like to adjust the default validation size (which will split the remaining data in half)?**")
  while True:
    choice = input("Enter 'y' or 'n': ").lower()
    if choice in ["y", "n"]:
      break
    else:
      print("Invalid choice. Please choose 'y' or 'n'.")

  if choice == "y":
    while True:
      try:
        validation_size = float(input(f"Enter the desired size for the validation set (between 0 and {remaining_data:.2f}): "))
        if 0 < validation_size < remaining_data:
          break
        else:
          print(f"Invalid input. Please enter a value between 0 and {remaining_data:.2f}.")
      except ValueError:
        print("Invalid input. Please enter a number.")
  else:
    validation_size = remaining_data / 2

  training_size = remaining_data - validation_size
  print(f"\n** You will use {test_size*100:.0f}% of your data for testing, {validation_size*100:.0f}% for validation, and {training_size*100:.0f}% for training.**")

  # Perform data splitting
  train_val, test = train_test_split(data, test_size=test_size, random_state=random_state)
  train, val = train_test_split(train_val, test_size=validation_size/(1-test_size), random_state=random_state)

  return train, val, test

def model_selection(train_data, val_data, test_data, target_column):
  """
  Guides the user through selecting a machine learning model for their task and preprocesses the data accordingly.

  Args:
      train_data (pandas.DataFrame): The DataFrame containing the training data.
      val_data (pandas.DataFrame): The DataFrame containing the validation data.
      test_data (pandas.DataFrame): The DataFrame containing the test data.
      target_column (str): The name of the column containing the target variable.

  Returns:
      tuple: (chosen_model, train_data, val_data, test_data)
  """

  print("** Model Selection is crucial for machine learning tasks.**")
  print("The chosen model should be suited to the type of problem you're trying to solve.")

  while True:
    chosen_model_type = input("\nEnter 'classification' or 'regression': ").lower()
    if chosen_model_type in ["classification", "regression"]:
      break
    else:
      print("Invalid choice. Please choose 'classification' or 'regression'.")

  if chosen_model_type == "classification":
    print("\n** Common Classification Models:**")
    print("- Logistic Regression (suitable for binary classification problems)")
    print("- Random Forest (powerful and versatile for various classification tasks)")
    print("** We will focus on Logistic Regression and Random Forest for now.**")
  else:
    print("\n** Common Regression Models:**")
    print("- Linear Regression (simple and interpretable for linear relationships)")
    print("- Random Forest (flexible for non-linear relationships)")
    print("** We will focus on Linear Regression and Random Forest for now.**")

  print("\n** Would you like to choose between the two models?**")
  print("(You can always try both models later!)")
  while True:
    choice = input("Enter 'y' or 'n': ").lower()
    if choice in ["y", "n"]:
      break
    else:
      print("Invalid choice. Please choose 'y' or 'n'.")

  if choice == "y":
    if chosen_model_type == "classification":
      while True:
        model_choice = input("Choose 'Logistic Regression' or 'Random Forest': ").lower()
        if model_choice in ["logistic regression", "random forest"]:
          chosen_model = LogisticRegression() if model_choice == "logistic regression" else RandomForestClassifier()
          break
        else:
          print("Invalid choice. Please choose 'Logistic Regression' or 'Random Forest'.")
    else:
      while True:
        model_choice = input("Choose 'Linear Regression' or 'Random Forest': ").lower()
        if model_choice in ["linear regression", "random forest"]:
          chosen_model = LinearRegression() if model_choice == "linear regression" else RandomForestClassifier()
          break
        else:
          print("Invalid choice. Please choose 'Linear Regression' or 'Random Forest'.")
  else:
    chosen_model = LogisticRegression() if chosen_model_type == "classification" else LinearRegression()

  # Encode categorical variables if it's a classification problem
  if chosen_model_type == "classification":
    categorical_columns = train_data.select_dtypes(include=['object']).columns.drop(target_column, errors='ignore')
    if len(categorical_columns) > 0:
      print("\n** Encoding categorical variables...**")
      onehot = OneHotEncoder(sparse=False, handle_unknown='ignore')
            
      for dataset in [train_data, val_data, test_data]:
        encoded_features = pd.DataFrame(onehot.fit_transform(dataset[categorical_columns]))
        encoded_features.columns = onehot.get_feature_names_out(categorical_columns)
        dataset.drop(columns=categorical_columns, inplace=True)
        dataset = pd.concat([dataset, encoded_features], axis=1)

      # Encode target variable if it's categorical
      if train_data[target_column].dtype == 'object':
        label_encoder = LabelEncoder()
        for dataset in [train_data, val_data, test_data]:
          dataset[target_column] = label_encoder.fit_transform(dataset[target_column])

  print(f"\n{type(chosen_model).__name__} model has been selected.")
  return chosen_model, train_data, val_data, test_data

def tune_hyperparameters(model_class, train_data, val_data, test_data, target_column, hyperparameter_grid):
  """
  Guides the user through hyperparameter tuning for a chosen machine learning model class.

  Args:
    model_class (object): The class of the machine learning model to be tuned.
    train_data (pandas.DataFrame): The DataFrame containing the training data (features and target).
    val_data (pandas.DataFrame): The DataFrame containing the validation data (features and target).
    test_data (pandas.DataFrame): The DataFrame containing the test data (features and target).
    target_column (str): The name of the target column.
    hyperparameter_grid (dict): A dictionary containing the hyperparameter grid for tuning.

  Returns:
    object: The best model found based on the hyperparameter tuning, or a default model if tuning is skipped.
  """
  from sklearn.base import is_classifier, is_regressor
  import warnings

  print(f"\n** You are working with the {model_class.__name__} model class.**")

  # Prompt user for hyperparameter tuning
  while True:
    print("\n** Hyperparameter tuning can significantly improve your model's performance.**")
    print("It involves trying different combinations of hyperparameter values and selecting the one that performs best on the validation data.")

    choice = input("Do you want to perform hyperparameter tuning? (y/n): ").lower()
    if choice in ['y', 'n']:
      break
    print("Invalid input. Please enter 'y' for yes or 'n' for no.")

  if choice == 'n':
    print("Skipping hyperparameter tuning. Using default model parameters.")
    return model_class()


  X_train = train_data.drop(target_column, axis=1)
  y_train = train_data[target_column]


  # Determine if the model is a classifier or regressor
  model_instance = model_class()
  if is_classifier(model_instance):
    scoring = 'accuracy'
    print("\n** Using accuracy as the scoring metric for classification.**")
  elif is_regressor(model_instance):
    scoring = 'neg_mean_squared_error'
    print("\n** Using negative mean squared error as the scoring metric for regression.**")
  else:
    scoring = 'accuracy'  # Default to accuracy if unsure
    print("\n** Unable to determine model type. Using accuracy as the default scoring metric.**")

  # Create the GridSearchCV object
  grid_search = GridSearchCV(model_instance, hyperparameter_grid, cv=5, scoring=scoring)

  convergence_warning = False
  with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")

    try:
      # Train the model with different hyperparameter combinations
      print("\n** Training the model with different hyperparameter combinations...**")
      grid_search.fit(X_train, y_train)

      # Display the best model and its parameters
      print("\n** The best model found based on validation performance:**")
      print(grid_search.best_estimator_)

      # Return the best model
      return grid_search.best_estimator_

    except Exception as e:
      # Handle convergence errors and suggest data scaling
      if "convergence" in str(e).lower() or "max_iter" in str(e).lower():
        print("\n**WARNING: Convergence issues encountered during hyperparameter tuning.**")
        print(f"Error message: {e}")
        print("\n** This might be due to the data not being scaled or normalized.**")
        print("Suggestions:")
        print("1. Scale/normalize your data before hyperparameter tuning.")
        print("2. Increase the max_iter parameter in the hyperparameter grid.")
        print("3. Try a different solver (e.g., 'liblinear' for smaller datasets, 'sag' or 'saga' for larger ones).")
        print("\nPlease rerun the program after applying these suggestions.")

      # Return a default model if an error occurred
      print("\nReturning a default model due to the encountered error.")
      return model_class()

  # This line should never be reached, but just in case:
  return model_class()

########################################################################################
# Model Training
########################################################################################
def train_model(chosen_model, train_data, val_data, test_data, target_column, hyperparameter_options=None):
  """
  Guides the user through training a machine learning model with interactive hyperparameter tuning (optional).

  Args:
      chosen_model (object): The machine learning model object to be trained.
      train_data (pandas.DataFrame): The DataFrame containing the training data (features and target).
      val_data (pandas.DataFrame): The DataFrame containing the validation data (features and target).
      test_data (pandas.DataFrame): The DataFrame containing the test data (features and target).
      hyperparameter_options (dict, optional): A dictionary containing options for hyperparameter tuning (e.g., learning rate, number of trees). Defaults to None.

  Returns:
      object: The trained machine learning model.
  """

  # Explain the purpose of model training
  print("** Model training is crucial for building a model that can learn from your data.**")
  print("The training process involves fitting the model to the training data, allowing it to identify patterns and relationships between features and the target variable.")

  # Hyperparameter Tuning (Optional)
  if hyperparameter_options is not None:
    print("\n** Hyperparameter tuning can significantly improve model performance.**")
    print("These are key parameters that control how the model learns. Would you like to explore some hyperparameter options?")
    while True:
      choice = input("Enter 'y' or 'n': ").lower()
      if choice in ["y", "n"]:
        break
      else:
        print("Invalid choice. Please choose 'y' or 'n'.")
    if choice == "y":
      # Integration Point
      tuned_model = tune_hyperparameters(chosen_model.__class__, train_data, val_data, hyperparameter_options)
      chosen_model = tuned_model  # Update chosen_model with the tuned version

  # Informative message about chosen model (**moved after Hyperparameter Tuning**)
  print(f"\n**You have chosen to train a {type(chosen_model).__name__} model.**")

  # Prepare the data
  X_train = train_data.drop(target_column, axis=1)
  y_train = train_data[target_column]



  # if isinstance(chosen_model, LogisticRegression):
    # chosen_model.set_params(max_iter=1000)

  # Train the model (**use the updated chosen_model**)
  print("\n**Training the model...**")
  trained_model = chosen_model.fit(X_train, y_train)

  print(f"\n{type(chosen_model).__name__} model was successfully trained!")

  # Return the trained model
  return trained_model



########################################################################################
# Model Evaluation
########################################################################################
def evaluate_model(trained_model, train_data, val_data, test_data, target_column):
  """
  Guides the user through basic model evaluation on the validation data, providing explanations for commonly used metrics.

  Args:
      trained_model (object): The trained machine learning model to be evaluated.
      val_data (pandas.DataFrame): The DataFrame containing the validation data (features and target).
  """

  print("\n** Evaluating the model's performance on the validation data...**")
  print("This helps us understand how well the model generalizes to unseen data.")
  X_val = val_data.drop(target_column, axis=1)
  y_val = val_data[target_column]


  # Make predictions on the validation data
  predictions = trained_model.predict(X_val)

  # Choose appropriate evaluation metrics based on the task (classification/regression)
  from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

  # Classification Task Example (assuming model predicts class labels)
  if not hasattr(trained_model, "predict_proba"):
    metric_value = accuracy_score(val_data[target_column], predictions)
  # For models with probability prediction capabilities (classification)
  else:
    # Choose appropriate metric based on task requirements (e.g., accuracy, precision, recall, F1)
    metric_value = accuracy_score(val_data[target_column], predictions)  # Replace with the most relevant metric

  # Informative message about the chosen metric
  if metric_value is not None:
    print(f"\n**Model performance on the validation data based on accuracy score: {metric_value:.4f} out of 1.0000**")  # Replace with metric name and formatting


  return metric_value, predictions  # Optional: Return the metric value for further analysis
