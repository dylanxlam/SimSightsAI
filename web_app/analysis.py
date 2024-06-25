########################################################################################
# Import Statements
########################################################################################
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib as plt
from sklearn.model_selection import learning_curve
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve
try:
  import shap  # Import shap library (assuming it's installed)
except ModuleNotFoundError:
  SHAP_AVAILABLE = False
else:
  SHAP_AVAILABLE = True
from sklearn.inspection import partial_dependence
from sklearn.inspection import permutation_importance
import pickle



########################################################################################
# Classification Reports
########################################################################################
def generate_classification_report(trained_model, val_data, predictions, target_column):
  """
  Guides the user through generating a classification report for the trained model on the validation data.

  Args:
      trained_model (object): The trained classification model.
      val_data (pandas.DataFrame): The DataFrame containing the validation data (features and target).
  """

  print("\n** Generating a classification report...**")
  print("This report provides a detailed breakdown of the model's performance for each class in the classification task.")




  # Generate and display the report
  report = classification_report(val_data[target_column], predictions, output_dict=True)
  print("\n** Classification Report:**")
  for class_name, metrics in report.items():
    print(f"\n** Class: {class_name} **")
    for metric_name, value in metrics.items(): 
      print(f"  - {metric_name}: {value:.4f}")  # Format metric values


  return report  # Optional: Return the report dictionary (if generated)


########################################################################################
# Confusion Matrices
########################################################################################
def visualize_confusion_matrix(trained_model, val_data):
  """
  Guides the user through generating and visualizing a confusion matrix for the trained model on the validation data.

  Args:
      trained_model (object): The trained classification model.
      val_data (pandas.DataFrame): The DataFrame containing the validation data (features and target).
  """

  print("\n** Visualizing the confusion matrix...**")
  print("This helps us understand how often the model correctly classified each class and where it made mistakes.")

  # Make predictions on the validation data
  predictions = trained_model.predict(val_data.drop("target", axis=1))

  # Confirmation for Confusion Matrix Visualization
  print("\n** Would you like to visualize the confusion matrix? (y/n) **")
  while True:
    choice = input().lower()
    if choice in ["y", "n"]:
      break
    else:
      print("Invalid choice. Please choose 'y' or 'n'.")

  if choice == "y":
    # Generate and display the confusion matrix (using library like seaborn)
    try:
      import seaborn as sns  # Import seaborn for visualization (optional)
      sns.heatmap(confusion_matrix(val_data["target"], predictions), annot=True, fmt="d")  # Annotate and format
      plt.show()  # Display the heatmap
    except ModuleNotFoundError:
      print("\n** seaborn library not found for visualization. Confusion matrix values:")
      print(confusion_matrix(val_data["target"], predictions))
    except Exception as e:
      print(f"\n** Error occurred while visualizing the confusion matrix: {e}")
    else:
      print("\n** Confusion matrix visualized (using seaborn if available).**")

  else:
    print("\n** Skipping confusion matrix visualization.**")


########################################################################################
# Learning Curves
########################################################################################   
def plot_learning_curves(trained_model, train_data, val_data):
  """
  Guides the user through plotting learning curves for the trained model.

  Args:
      trained_model (object): The trained classification or regression model.
      train_data (pandas.DataFrame): The DataFrame containing the training data (features and target).
      val_data (pandas.DataFrame): The DataFrame containing the validation data (features and target).
  """

  print("\n** Visualizing learning curves...**")
  print("Learning curves show how the model's performance (training and validation scores) changes with the amount of training data.")

  # Confirmation for Learning Curve Plot
  print("\n** Would you like to visualize learning curves? (y/n) **")
  while True:
    choice = input().lower()
    if choice in ["y", "n"]:
      break
    else:
      print("Invalid choice. Please choose 'y' or 'n'.")

  if choice == "y":

    # Extract features and target
    X_train = train_data.drop("target", axis=1)
    y_train = train_data["target"]
    X_val = val_data.drop("target", axis=1)
    y_val = val_data["target"]

    # Define model class (assuming you have the class definition)
    model_class = trained_model.__class__  # Get the class of the trained model

    # Define train_sizes for the curve
    train_sizes, train_scores, val_scores = learning_curve(model_class(), X_train, y_train, cv=5, scoring="accuracy")  # Replace 'accuracy' with relevant metric

    # Plot the learning curve
    plt.figure()
    plt.plot(train_sizes, train_scores.mean(axis=1), label="Training Score")
    plt.plot(train_sizes, val_scores.mean(axis=1), label="Validation Score")
    plt.ylabel("Score")
    plt.xlabel("Training Set Size")
    plt.title("Learning Curves")
    plt.legend()
    plt.grid(True)
    plt.show()

  else:
    print("\n** Skipping learning curves visualization.**")


########################################################################################
# ROC Curves (for Classification)
########################################################################################
def plot_roc_curve(trained_model, val_data):
  """
  Guides the user through plotting ROC curves for the trained classification model.

  Args:
      trained_model (object): The trained classification model.
      val_data (pandas.DataFrame): The DataFrame containing the validation data (features and target).
  """

  print("\n** Visualizing ROC curves (for classification models only)...**")
  print("ROC curves show the trade-off between true positive rates (TPR) and false positive rates (FPR) at different classification thresholds.")

  # Check if the model is a classification model
  if not hasattr(trained_model, "predict_proba"):
    return

  # Confirmation for ROC Curve Plot
  print("\n** Would you like to visualize ROC curves? (y/n) **")
  while True:
    choice = input().lower()
    if choice in ["y", "n"]:
      break
    else:
      print("Invalid choice. Please choose 'y' or 'n'.")

  if choice == "y":
    # Import libraries for plotting (replace with your preferred library)
    try:
      import matplotlib.pyplot as plt
    except ModuleNotFoundError as e:
      print(f"\n** Required library (matplotlib) not found for plotting. Skipping ROC curve.**")
      return

    # Extract features and target
    X_val = val_data.drop("target", axis=1)
    y_val = val_data["target"]

    # Predict probabilities
    y_pred_proba = trained_model.predict_proba(X_val)[:, 1]  # Assuming positive class probability

    # Calculate ROC curve metrics
    fpr, tpr, thresholds = roc_curve(y_val, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    # Plot the ROC curve
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', label='No Skill')
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True)
    plt.show()

  else:
    print("\n** Skipping ROC curve visualization.**")


########################################################################################
# Precision-Recall Curves (for Classification)
########################################################################################
def plot_precision_recall_curve(trained_model, val_data):
  """
  Guides the user through plotting precision-recall curves for the trained classification model.

  Args:
      trained_model (object): The trained classification model.
      val_data (pandas.DataFrame): The DataFrame containing the validation data (features and target).
  """

  print("\n** Visualizing precision-recall curves (for classification models only)...**")
  print("Precision-recall curves show the trade-off between precision (minimizing false positives) and recall (minimizing false negatives) at different classification thresholds.")

  # Check if the model is a classification model
  if not hasattr(trained_model, "predict_proba"):
    print("\n** This function is only applicable for classification models with probability prediction capabilities (predict_proba). Skipping precision-recall curve.**")
    return

  # Confirmation for Precision-Recall Curve Plot
  print("\n** Would you like to visualize precision-recall curves? (y/n) **")
  while True:
    choice = input().lower()
    if choice in ["y", "n"]:
      break
    else:
      print("Invalid choice. Please choose 'y' or 'n'.")

  if choice == "y":
    # Import libraries for plotting (replace with your preferred library)
    try:
      import matplotlib.pyplot as plt
    except ModuleNotFoundError as e:
      print(f"\n** Required library (matplotlib) not found for plotting. Skipping precision-recall curve.**")
      return

    # Extract features and target
    X_val = val_data.drop("target", axis=1)
    y_val = val_data["target"]

    # Predict probabilities
    y_pred_proba = trained_model.predict_proba(X_val)[:, 1]  # Assuming positive class probability

    # Calculate precision-recall curve metrics
    precision, recall, thresholds = precision_recall_curve(y_val, y_pred_proba)

    # Plot the precision-recall curve
    plt.figure()
    plt.plot(recall, precision, label='Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True)
    plt.show()

  else:
    print("\n** Skipping precision-recall curve visualization.**")


########################################################################################
# SHAP Explanations
########################################################################################
def explain_with_shap(trained_model, val_data, explainer_type="force_plot"):
  """
  Guides the user through generating SHAP explanations for the trained model (if shap library is available).

  Args:
      trained_model (object): The trained model (any model supported by shap).
      val_data (pandas.DataFrame): The DataFrame containing the validation data (features and target).
      explainer_type (str, optional): The type of SHAP explainer to use (e.g., "force_plot", "tree_explainer"). Defaults to "force_plot".
  """

  if not SHAP_AVAILABLE:
    print("\n** SHAP library not found. SHAP explanations are unavailable.**")
    return

  print("\n** Generating SHAP explanations (may take some time)...**")
  print("SHAP explains how each feature contributes to a model's prediction for a specific data point.")

  # Confirmation for SHAP Explanations
  print("\n** Would you like to generate SHAP explanations? (y/n) **")
  while True:
    choice = input().lower()
    if choice in ["y", "n"]:
      break
    else:
      print("Invalid choice. Please choose 'y' or 'n'.")

  if choice == "y":
    # Extract features and a single data point for explanation (can be modified to explain multiple points)
    X_val = val_data.drop("target", axis=1)
    # Choose a data point for explanation (e.g., first row)
    instance = X_val.iloc[0].values.reshape(1, -1)  # Reshape for single instance

    # Create a SHAP explainer (consider different explainers based on model type)
    if explainer_type == "force_plot":
      explainer = shap.Explainer(trained_model.predict, instance)  # Force plot explainer
    elif explainer_type == "tree_explainer" and hasattr(trained_model, "tree_"):
      explainer = shap.TreeExplainer(trained_model)  # Tree explainer for tree-based models (if applicable)
    else:
      print(f"\n** Unsupported explainer type: {explainer_type}. Using force_plot explainer.")
      explainer = shap.Explainer(trained_model.predict, instance)  # Fallback to force plot

    # Generate SHAP explanation
    shap_values = explainer(instance)

    # Display SHAP explanation (using shap library functions)
    try:
      shap.force_plot(explainer.base_value, instance, shap_values, feature_names=X_val.columns)  # Force plot for any model
    except:
      if explainer_type == "tree_explainer":
        shap.summary_plot(shap_values, instance, feature_names=X_val.columns)  # Tree explainer summary plot (if applicable)
      else:
        print("\n** Unable to display SHAP explanation using the chosen explainer type. Try 'force_plot'.")

  else:
    print("\n** Skipping SHAP explanations.**")


########################################################################################
# Partial Dependence Plots (PDPs)
########################################################################################
def plot_partial_dependence(trained_model, val_data, feature_names=None):
  """
  Guides the user through generating and visualizing partial dependence plots (PDPs) for the trained model.

  Args:
      trained_model (object): The trained model (any model supported by sklearn.inspection.partial_dependence).
      val_data (pandas.DataFrame): The DataFrame containing the validation data (features and target).
      feature_names (list, optional): The list of feature names (in the same order as the data). Defaults to None.
  """

  print("\n** Visualizing partial dependence plots (PDPs)...**")
  print("PDPs show the average effect of a single feature on the model's prediction, marginalizing over other features.")

  # Confirmation for PDP Visualization
  print("\n** Would you like to visualize PDPs? (y/n) **")
  while True:
    choice = input().lower()
    if choice in ["y", "n"]:
      break
    else:
      print("Invalid choice. Please choose 'y' or 'n'.")

  if choice == "y":
    # Extract features and target
    X_val = val_data.drop("target", axis=1)
    y_val = val_data["target"]

    # Get feature names if not provided
    if feature_names is None:
      feature_names = X_val.columns.tolist()

    # Choose a feature for PDP (can be modified to plot multiple features)
    print("\n** Select a feature to visualize its PDP: ")
    for i, feature_name in enumerate(feature_names):
      print(f"{i+1}. {feature_name}")
    while True:
      try:
        feature_index = int(input()) - 1
        if 0 <= feature_index < len(feature_names):
          break
        else:
          print("Invalid choice. Please choose a feature number between 1 and", len(feature_names))
      except ValueError:
        print("Invalid input. Please enter a number.")

    # Calculate PDP using partial_dependence
    pdp, residuals = partial_dependence(trained_model, X_val, features=[feature_index])

    # Plot the PDP
    try:
      import matplotlib.pyplot as plt
      plt.figure()
      plt.plot(X_val.iloc[:, feature_index], pdp.ravel(), label=feature_names[feature_index])  # Assuming single feature
      plt.xlabel(feature_names[feature_index])
      plt.ylabel("Average Prediction")
      plt.title("Partial Dependence Plot")
      plt.legend()
      plt.grid(True)
      plt.show()
    except ModuleNotFoundError as e:
      print(f"\n** Required library (matplotlib) not found for plotting. Skipping PDP.**")
      return

  else:
    print("\n** Skipping partial dependence plot visualization.**")


########################################################################################
# Feature Importance Analysis
########################################################################################
def analyze_feature_importance(trained_model, val_data, feature_names=None):
  """
  Guides the user through analyzing feature importance for the trained model.

  Args:
      trained_model (object): The trained model (any model supported by sklearn.inspection.permutation_importance).
      val_data (pandas.DataFrame): The DataFrame containing the validation data (features and target).
      feature_names (list, optional): The list of feature names (in the same order as the data). Defaults to None.
  """

  print("\n** Analyzing feature importance...**")
  print("Feature importance helps identify features that have a greater impact on the model's predictions.")

  # Confirmation for Feature Importance Analysis
  print("\n** Would you like to analyze feature importance? (y/n) **")
  while True:
    choice = input().lower()
    if choice in ["y", "n"]:
      break
    else:
      print("Invalid choice. Please choose 'y' or 'n'.")

  if choice == "y":
    # Extract features and target
    X_val = val_data.drop("target", axis=1)
    y_val = val_data["target"]

    # Get feature names if not provided
    if feature_names is None:
      feature_names = X_val.columns.tolist()

    # Calculate feature importance using permutation_importance
    results = permutation_importance(trained_model, X_val, y_val, scoring="accuracy")  # Replace 'accuracy' with relevant metric

    # Print feature importances (sorted by importance)
    feature_importances = results.importances_mean
    feature_importances_sorted = sorted(zip(feature_names, feature_importances), key=lambda x: x[1], reverse=True)
    print("\n** Feature Importance Ranking: **")
    for i, (feature_name, importance) in enumerate(feature_importances_sorted):
      print(f"{i+1}. {feature_name}: {importance:.4f}")

  else:
    print("\n** Skipping feature importance analysis.**")


########################################################################################
# Model Persistence
########################################################################################
def save_model(trained_model, save_path):
  """
  Guides the user through saving the trained model.

  Args:
      trained_model (object): The trained model to save.
      save_path (str): The path (including filename) to save the model.
  """

  print("\n** Saving the trained model (optional)...**")
  print("This allows you to use the trained model for future predictions without retraining.")

  # Confirmation for Model Persistence
  print("\n** Would you like to save the trained model? (y/n) **")
  while True:
    choice = input().lower()
    if choice in ["y", "n"]:
      break
    else:
      print("Invalid choice. Please choose 'y' or 'n'.")

  if choice == "y":
    try:
      # Save the model using pickle
      with open(save_path, 'wb') as f:
        pickle.dump(trained_model, f)
      print(f"\n** Model saved successfully to: {save_path}**")
    except Exception as e:
      print(f"\n** Error saving the model: {e}**")

  else:
    print("\n** Skipping model persistence.**")
