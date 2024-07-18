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


  print("\nWe will start by splitting your data into training, validation, and test sets.")
  print("\nData Splitting is crucial for training and evaluating machine learning models.")
  print("It separates your data into three sets: training, validation, and test.")
  print("The training set is used to build the model, the validation set is used to assess its performance on")
  print("unseen data during training (hyperparameter tuning), and the test set provides a final evaluation on")
  print("completely unseen data after training is complete.\n")

  print(f"\nBy default, we will use {test_size*100:.0f}% of your data for testing, and the remaining data will")
  print("be split between training and validation sets using scikit-learn's train_test_split function.")
  print(f"\nWould you like to adjust the default test set size (currently {test_size:.2f})?")
  while True:
    choice = input("Enter 'y' or 'n': ").lower()
    if choice in ["y", "n"]:
      break
    else:
      print("\nInvalid choice. Please choose 'y' or 'n'.")
  if choice == "y":
    while True:
      try:
        test_size = float(input("Enter the desired size for the test set (between 0 and 1): "))
        if 0 < test_size < 1:
          break
        else:
          print("\nInvalid input. Please enter a value between 0 and 1.")
      except ValueError:
        print("\nInvalid input. Please enter a number.")

  remaining_data = 1 - test_size
  print(f"\nAfter allocating {test_size*100:.0f}% for testing, you have {remaining_data*100:.0f}% of data remaining for training and validation.")
  print("Would you like to adjust the default validation size (which will split the remaining data in half?")
  while True:
    choice = input("Enter 'y' or 'n': ").lower()
    if choice in ["y", "n"]:
      break
    else:
      print("\nInvalid choice. Please choose 'y' or 'n'.")

  if choice == "y":
    while True:
      try:
        validation_size = float(input(f"Enter the desired size for the validation set (between 0 and {remaining_data:.2f}): "))
        if 0 < validation_size < remaining_data:
          break
        else:
          print(f"\nInvalid input. Please enter a value between 0 and {remaining_data:.2f}.")
      except ValueError:
        print("\nInvalid input. Please enter a number.")
  else:
    validation_size = remaining_data / 2

  training_size = remaining_data - validation_size
  print(f"\n\n\nYou will use {test_size*100:.0f}% of your data for testing, {validation_size*100:.0f}% for validation, and {training_size*100:.0f}% for training.")

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

  print("\n\n\nNow that you have training, validation, and testing data, we can move onto selecting a machine learning model!")

  print("\nSimply put, classification models allow for prediction of a categorical variable, and regression models allow for prediction of a numerical variable.")
  print("Identify if your target variable is a numerical or categorical variable then choose:")
  while True:
    chosen_model_type = input("\'classification' or 'regression': ").lower()
    if chosen_model_type in ["classification", "regression"]:
      break
    else:
      print("\nInvalid choice. Please choose 'classification' or 'regression'.")

  if chosen_model_type == "classification":
    print("\n\n\nCommon Classification Models:")
    print("- Logistic Regression (suitable for binary classification problems)")
    print("- Random Forest (powerful and versatile for various classification tasks)")
    print("\nThe default model for your model type is Logistic Regression. Would you like to choose between the two models?")

  else:
    print("\n\n\nCommon Regression Models:")
    print("- Linear Regression (simple and interpretable for linear relationships)")
    print("- Random Forest (flexible for non-linear relationships)")
    print("\nThe default model for your model type is Logistic Regression. Would you like to choose between the two models?")

  print("(You can always try both models later!)")
  while True:
    choice = input("Enter 'y' or 'n': ").lower()
    if choice in ["y", "n"]:
      break
    else:
      print("\nInvalid choice. Please choose 'y' or 'n'.")

  if choice == "y":
    if chosen_model_type == "classification":
      while True:
        model_choice = input("Choose 'Logistic Regression' or 'Random Forest': ").lower()
        if model_choice in ["logistic regression", "random forest"]:
          chosen_model = LogisticRegression() if model_choice == "logistic regression" else RandomForestClassifier()
          break
        else:
          print("\nInvalid choice. Please choose 'Logistic Regression' or 'Random Forest'.")
    else:
      while True:
        model_choice = input("Choose 'Linear Regression' or 'Random Forest': ").lower()
        if model_choice in ["linear regression", "random forest"]:
          chosen_model = LinearRegression() if model_choice == "linear regression" else RandomForestClassifier()
          break
        else:
          print("\nInvalid choice. Please choose 'Linear Regression' or 'Random Forest'.")
  else:
    chosen_model = LogisticRegression() if chosen_model_type == "classification" else LinearRegression()

  # Encode categorical variables if it's a classification problem
  if chosen_model_type == "classification":
    categorical_columns = train_data.select_dtypes(include=['object']).columns.drop(target_column, errors='ignore')
    if len(categorical_columns) > 0:
      print("\nEncoding categorical variables is suggested for classification problems. Ensuring categorical variables are encoded...")
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
    """
    from sklearn.base import is_classifier, is_regressor
    import warnings
    import pandas as pd

    print(f"You are working with the {model_class.__name__} model class.")
    print("\n\n\nNow that you have chosen your model, we will now move onto hyperparameter tuning!\n")
    
    while True:
        print("\nHyperparameter tuning can significantly improve your model's performance.")
        print("It involves trying different combinations of hyperparameter values and selecting the one that performs best on the validation data.")

        choice = input("Do you want to perform hyperparameter tuning? (y/n): ").lower()
        if choice in ['y', 'n']:
            break
        print("\nInvalid input. Please enter 'y' for yes or 'n' for no.")

    if choice == 'n':
        print("\nSkipping hyperparameter tuning. Using default model parameters.")
        return model_class()

    X_train = train_data.drop(target_column, axis=1)
    y_train = train_data[target_column]

    # Determine if the model is a classifier or regressor
    model_instance = model_class()
    if is_classifier(model_instance):
        scoring = 'accuracy'
        print("\nSince we are using a classification model, we will use accuracy as the scoring metric for classification.")
    elif is_regressor(model_instance):
        scoring = 'neg_mean_squared_error'
        print("\nSince we are using a regression model, we will use negative mean squared error as the scoring metric for regression.")
    else:
        scoring = 'accuracy'  # Default to accuracy if unsure
        print("\nUnable to determine model type. Using accuracy as the default scoring metric.")

    # Create the GridSearchCV object
    grid_search = GridSearchCV(model_instance, hyperparameter_grid, cv=5, scoring=scoring)

    convergence_warning = False
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")


        try:
            # Train the model with different hyperparameter combinations
            print("\nTraining the model with different hyperparameter combinations...")
            grid_search.fit(X_train, y_train)

            # Display the best model and its parameters
            print("  - The best model found based on validation performance:")
            print("    ", grid_search.best_estimator_)

            # Return the best model and the original data
            return grid_search.best_estimator_, train_data, val_data, test_data

        except ValueError as e:
            if "could not convert string to float" in str(e):
                print("\nDetected non-numeric data. Applying label encoding to categorical features.")
                
                # Merge all datasets
                all_data = pd.concat([train_data, val_data, test_data], axis=0)
                
                # Identify categorical columns
                categorical_cols = all_data.select_dtypes(include=['object']).columns.tolist()
                
                # Apply label encoding to all data
                for col in categorical_cols:
                    le = LabelEncoder()
                    all_data[col] = le.fit_transform(all_data[col].astype(str))
                
                # Split back into train, val, and test
                train_data = all_data[:len(train_data)]
                val_data = all_data[len(train_data):len(train_data)+len(val_data)]
                test_data = all_data[len(train_data)+len(val_data):]
                
                print("\nLabel encoding applied. Restarting hyperparameter tuning with encoded data.")
                
                # Retry hyperparameter tuning with encoded data
                X_train = train_data.drop(target_column, axis=1)
                y_train = train_data[target_column]
                grid_search.fit(X_train, y_train)
                
                print("  - The best model found based on validation performance:")
                print("    ", grid_search.best_estimator_)
                
                return grid_search.best_estimator_, train_data, val_data, test_data


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
  print("\n\n\nNow that you have chosen your model parameters, we will move on to train your model!")
  print("The training process involves fitting the model to the training data, allowing it to identify patterns and")
  print("relationships between features and the target variable.")


  # Informative message about chosen model (moved after Hyperparameter Tuning)
  print(f"\nWe will now train your {type(chosen_model).__name__} model.")

  # Prepare the data
  X_train = train_data.drop(target_column, axis=1)
  y_train = train_data[target_column]

  # Train the model (use the updated chosen_model)
  print("Training the model...")
  trained_model = chosen_model.fit(X_train, y_train)

  print(f"  - {type(chosen_model).__name__} model was successfully trained!")

  # Return the trained model
  return trained_model



########################################################################################
# Model Evaluation
########################################################################################
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def evaluate_model(trained_model, train_data, val_data, test_data, target_column):
    """
    Guides the user through basic model evaluation on the validation data, providing explanations for commonly used metrics.

    Args:
        trained_model (object): The trained machine learning model to be evaluated.
        train_data (pandas.DataFrame): The DataFrame containing the training data.
        val_data (pandas.DataFrame): The DataFrame containing the validation data.
        test_data (pandas.DataFrame): The DataFrame containing the test data.
        target_column (str): The name of the target column.
    """
    print("\n\n\nGreat! Now your model is trained, we will now move on to evaluate your model's performance.")
    print("This helps us understand how well the model generalizes to unseen data.")
    print("\nEvaluating the model's performance on the validation data...")
    
    X_val = val_data.drop(target_column, axis=1)
    y_val = val_data[target_column]

    # Make predictions on the validation data
    predictions = trained_model.predict(X_val)

    # Determine if it's a classification or regression task
    is_classification = hasattr(trained_model, "predict_proba") or len(np.unique(y_val)) <= 10

    if is_classification:
        # Classification metrics
        accuracy = accuracy_score(y_val, predictions)
        precision = precision_score(y_val, predictions, average='weighted')
        recall = recall_score(y_val, predictions, average='weighted')
        f1 = f1_score(y_val, predictions, average='weighted')

        print(f"  - Accuracy: {accuracy:.4f}")
        print(f"  - Precision: {precision:.4f}")
        print(f"  - Recall: {recall:.4f}")
        print(f"  - F1 Score: {f1:.4f}")

        print("\nHow to interpret These Metrics:")
        print("  - Accuracy: Proportion of correct predictions (both true positives and true negatives) among the total number of cases examined.")
        print("  - Precision: Ability of the classifier not to label as positive a sample that is negative.")
        print("  - Recall: Ability of the classifier to find all the positive samples.")
        print("  - F1 Score: The harmonic mean of precision and recall, providing a single score that balances both concerns.")

        return accuracy, predictions

    else:
        # Regression metrics
        mse = mean_squared_error(y_val, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_val, predictions)
        r2 = r2_score(y_val, predictions)

        print(f"  - Mean Squared Error (MSE): {mse:.4f}")
        print(f"  - Root Mean Squared Error (RMSE): {rmse:.4f}")
        print(f"  - Mean Absolute Error (MAE): {mae:.4f}")
        print(f"  - R-squared (R2) Score: {r2:.4f}")

        print("\nHow to Interpret These Metrics:")
        print("  - MSE: Average squared difference between the estimated values and the actual value.")
        print("  - RMSE: Square root of MSE, giving a measure of the average magnitude of the error in the same units as the target variable.")
        print("  - MAE: Average absolute difference between the estimated values and the actual value.")
        print("  - R-squared: Proportion of the variance in the dependent variable that is predictable from the independent variable(s).")

        return r2, predictions

