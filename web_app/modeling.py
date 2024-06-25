########################################################################################
# Import Statements
########################################################################################
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder





########################################################################################
# Model Selection
########################################################################################
def model_selection(data, target_column):
  """
  Guides the user through selecting a machine learning model for their task.

  Args:
      data (pandas.DataFrame): The DataFrame containing the data.
      target_col (str): The name of the column containing the target variable.

  Returns:
      str: The name of the chosen machine learning model.
  """

  # Explain the purpose of model selection
  print("** Model Selection is crucial for machine learning tasks.**")
  print("The chosen model should be suited to the type of problem you're trying to solve.")

  # User input for problem type
  print("\n** What kind of problem are you trying to solve?**")
  print("- Classification (predict a category, e.g., spam or not spam)")
  print("- Regression (predict a continuous value, e.g., house price)")

  while True:
    chosen_model = input("Enter 'classification' or 'regression': ").lower()
    if chosen_model in ["classification", "regression"]:
      break
    else:
      print("Invalid choice. Please choose 'classification' or 'regression'.")

  # Suggest models based on problem type
  if chosen_model == "classification":
    print("\n** Common Classification Models:**")
    print("- Logistic Regression (suitable for binary classification problems)")
    print("- Random Forest (powerful and versatile for various classification tasks)")
    print("- Support Vector Machines (effective for high-dimensional data)")
    print("** We will focus on Logistic Regression and Random Forest for now.**")
  else:
    print("\n** Common Regression Models:**")
    print("- Linear Regression (simple and interpretable for linear relationships)")
    print("- Decision Tree Regression (flexible for non-linear relationships)")
    print("- Support Vector Regression (effective for handling outliers)")
    print("** We will focus on Linear Regression and Decision Tree Regression for now.**")

  # User confirmation for model choice (optional)
  if chosen_model == "classification":
    print("\n** Would you like to choose between Logistic Regression and Random Forest?**")
    print("(You can always try both models later!)")
    while True:
      choice = input("Enter 'y' or 'n': ").lower()
      if choice in ["y", "n"]:
        break
      else:
        print("Invalid choice. Please choose 'y' or 'n'.")
    if choice == "y":
      print("\n** Briefly:")
      print("- Logistic Regression: Good for binary (e.g. yes or no, 0 or 1) classification, interpretable results.")
      print("- Random Forest: More powerful, handles complex relationships better.")
      while True:
        model_choice = input("Choose 'Logistic Regression' or 'Random Forest': ").lower()
        if model_choice in ["logistic regression", "random forest"]:
          if model_choice == "logistic regression":
            return LogisticRegression()
          else:
            return RandomForestClassifier()
        else:
          print("Invalid choice. Please choose 'Logistic Regression' or 'Random Forest'.")
    else:
      # Default selection
      return LogisticRegression()
  else:
    if choice == "y":
      while True:
        model_choice = input("Choose 'Linear Regression' or 'Random Forest': ").lower()
        if model_choice in ["linear regression", "decision tree regression"]:
          if model_choice == "linear regression":
            return LinearRegression()  # Return the model object
          else:
            return RandomForestClassifier()  # Return the model object
        else:
          print("Invalid choice. Please choose 'Linear Regression' or 'Decision Tree Regression'.")
    else:
      # Default selection (e.g., LinearRegression())
      return LinearRegression()  # Return the default model object


########################################################################################
# Data Splitting
########################################################################################
def data_splitting(data, target_column, test_size=0.2, random_state=42):
  """
  Guides the user through splitting data into training, validation, and test sets 
  for machine learning tasks, allowing adjustment of training size.

  Args:
      data (pandas.DataFrame): The DataFrame containing the data.
      target_col (str): The name of the column containing the target variable.
      test_size (float, optional): The desired size for the test set (between 0 and 1, default: 0.2).
      random_state (int, optional): Sets the random seed for reproducibility (default: 42).

  Returns:
      tuple: A tuple containing the training, validation, and test DataFrames.
  """

  # Explain the purpose of data splitting
  print("** Data Splitting is crucial for training and evaluating machine learning models.**")
  print("It separates your data into three sets: training, validation, and test.")
  print("The training set is used to build the model, the validation set is used to assess its performance on unseen data during training (hyperparameter tuning), and the test set provides a final evaluation on completely unseen data after training is complete.")

  # Informative message about split size (can be adjusted for validation size)
  print(f"\n** By default, we will use {test_size*100:.0f}% of your data for testing, and the remaining data will be split between training and validation sets using scikit-learn's train_test_split function.**")
  print("** Would you like to adjust the default test set size (currently {test_size:.2f})?**")
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
        if 0 <= test_size <= 1:
          break
        else:
          print("Invalid input. Please enter a value between 0 and 1.")
      except ValueError:
        print("Invalid input. Please enter a number.")

  # Informative message about remaining data for training/validation
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
        validation_size = float(input("Enter the desired size for the validation set (between 0 and 1, and less than the remaining data): "))
        if 0 <= validation_size <= remaining_data and validation_size < 1:
          training_size = remaining_data - validation_size
          break
        else:
          print("Invalid input. Please enter a value between 0 and", remaining_data, "and less than 1.")
      except ValueError:
        print("Invalid input. Please enter a number.")
  else:
    # Default behavior: validation size = half of remaining data
    validation_size = remaining_data / 2
    training_size = remaining_data / 2

  # Informative message about final split percentages
  print(f"\n** You will use {test_size*100:.0f}% of your data for testing, {validation_size*100:.0f}% for validation, and {training_size*100:.0f}% for training.**")



  X = data.drop(target_column, axis=1)  # Separate features (X) and target (y)
  y = data[target_column]

  # Perform data splitting based on chosen sizes
  X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

  # Further split training/validation sets based on user choice
  if choice == "y":  # User wants to adjust validation size
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=validation_size, random_state=random_state)
  else:  # Use default validation size (half of remaining data)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=validation_size, random_state=random_state)

  # Combine features and target back into DataFrames
  X_train_df = pd.DataFrame(X_train, columns=X.columns)
  X_val_df = pd.DataFrame(X_val, columns=X.columns)
  X_test_df = pd.DataFrame(X_test, columns=X.columns)
  y_train_df = pd.Series(y_train, name=target_column)
  y_val_df = pd.Series(y_val, name=target_column)
  y_test_df = pd.Series(y_test, name=target_column)

  # Combine features and target into DataFrames
  train_data = pd.concat([X_train_df, y_train_df], axis=1)
  val_data = pd.concat([X_val_df, y_val_df], axis=1)
  test_data = pd.concat([X_test_df, y_test_df], axis=1)

  # Return the split DataFrames
  return train_data, val_data, test_data


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

  categorical_columns = X_train.select_dtypes(include=['object']).columns

  # Encode categorical variables
  if len(categorical_columns) > 0:
    print("\n** Verifying that categorical variables are encoded...**")
    X_train = pd.get_dummies(X_train, columns=categorical_columns)

    # Ensure val_data and test_data have the same columns as X_train
    X_val = pd.get_dummies(val_data.drop(target_column, axis=1), columns=categorical_columns)

    # Add missing columns to val and test data
    for col in X_train.columns:
      if col not in X_val.columns:
        X_val[col] = 0

    X_val = X_val[X_train.columns]

    if y_train.dtype == 'object':
      label_encoder = LabelEncoder()
      y_train = label_encoder.fit_transform(y_train)
      y_val = label_encoder.transform(val_data[target_column])
    else:
      y_val = val_data[target_column]


    if isinstance(chosen_model, LogisticRegression):
      chosen_model.set_params(max_iter=1000)

  # Train the model (**use the updated chosen_model**)
  print("\n**Training the model...**")
  trained_model = chosen_model.fit(X_train, y_train)

  print(f"\n{type(chosen_model).__name__} model was successfully trained!")

  # Return the trained model
  return trained_model


########################################################################################
# Hyperparameter Tuning
########################################################################################
def tune_hyperparameters(model_class, train_data, val_data, test_data, target_column, hyperparameter_grid):
  """
  Guides the user through hyperparameter tuning for a chosen machine learning model class.

  Args:
      model_class (object): The class of the machine learning model to be tuned (e.g., scikit-learn's RandomForestClassifier).
      train_data (pandas.DataFrame): The DataFrame containing the training data (features and target).
      val_data (pandas.DataFrame): The DataFrame containing the validation data (features and target).
      hyperparameter_grid (dict): A dictionary containing the hyperparameter grid for tuning (e.g., {"n_estimators": [100, 200], "max_depth": [3, 5]}).

  Returns:
      object: The best model found based on the hyperparameter tuning.
  """

  # Explain the purpose of hyperparameter tuning
  print("\n** Hyperparameter tuning can significantly improve your model's performance.**")
  print("It involves trying different combinations of hyperparameter values and selecting the one that performs best on the validation data.")

  # Informative message about chosen model class
  print(f"\n** You are tuning hyperparameters for the {model_class.__name__} model class.**")


  X_train = train_data.drop(target_column, axis=1)
  y_train = train_data[target_column]


  categorical_columns = X_train.select_dtypes(include=['object']).columns

  # Encode categorical variables
  if len(categorical_columns) > 0:
    print("\n** Verifying that categorical variables are encoded...**")
    X_train = pd.get_dummies(X_train, columns=categorical_columns)

    # Ensure val_data and test_data have the same columns as X_train
    X_val = pd.get_dummies(val_data.drop(target_column, axis=1), columns=categorical_columns)

    # Add missing columns to val and test data
    for col in X_train.columns:
      if col not in X_val.columns:
        X_val[col] = 0

    X_val = X_val[X_train.columns]

    if y_train.dtype == 'object':
      label_encoder = LabelEncoder()
      y_train = label_encoder.fit_transform(y_train)
      y_val = label_encoder.transform(val_data[target_column])
    else:
      y_val = val_data[target_column]

  # Create the GridSearchCV object
  grid_search = GridSearchCV(model_class(), hyperparameter_grid, cv=5, scoring="accuracy", max_iter=1000)  # Replace 'accuracy' with appropriate metric



  # Train the model with different hyperparameter combinations
  print("\n** Training the model with different hyperparameter combinations...**")
  grid_search.fit(X_train, y_train)

  # Display the best model and its parameters
  print("\n** The best model found based on validation performance:**")
  print(grid_search.best_estimator_)

  # Return the best model
  return grid_search.best_estimator_


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
  X_train = train_data.drop(target_column, axis=1)
  y_train = train_data[target_column]

  categorical_columns = X_train.select_dtypes(include=['object']).columns

  # Encode categorical variables
  if len(categorical_columns) > 0:
    print("\n** Verifying that categorical variables are encoded...**")
    X_train = pd.get_dummies(X_train, columns=categorical_columns)

    # Ensure val_data and test_data have the same columns as X_train
    X_val = pd.get_dummies(val_data.drop(target_column, axis=1), columns=categorical_columns)

    # Add missing columns to val and test data
    for col in X_train.columns:
      if col not in X_val.columns:
        X_val[col] = 0

    X_val = X_val[X_train.columns]



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
