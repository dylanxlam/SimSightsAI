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
    if isinstance(metrics, dict):
      print(f"\n** Class: {class_name} **")
      for metric_name, value in metrics.items(): 
        print(f"  - {metric_name}: {value:.4f}")  # Format metric values
    else:
      # Handle the 'accuracy' metric separately
      print(f"\n** Overall Accuracy: {metrics:.4f} **")

  return report  # Optional: Return the report dictionary (if generated)


########################################################################################
# Confusion Matrices
########################################################################################
def visualize_confusion_matrix(trained_model, val_data, predictions, target_column):
  """
  Guides the user through generating and visualizing a confusion matrix for the trained model on the validation data.

  Args:
      trained_model (object): The trained classification model.
      val_data (pandas.DataFrame): The DataFrame containing the validation data (features and target).
  """



  # Confirmation for Confusion Matrix Visualization
  print("\nConfusion matrices help us understand how often the model correctly classified each class and where it made mistakes.")
  print("\n** Would you like to visualize the confusion matrix? (y/n) **")

  while True:
    choice = input().lower()
    if choice in ["y", "n"]:
      break
    else:
      print("Invalid choice. Please choose 'y' or 'n'.")

  if choice == "y":
    # Generate and display the confusion matrix (using library like seaborn)
    print("\n** Visualizing the confusion matrix...**")

    try:
      import seaborn as sns  # Import seaborn for visualization (optional)
      import matplotlib.pyplot as plt  # Explicitly import matplotlib for plt.show()
      print("Close the plot window to continue with this program.")
      sns.heatmap(confusion_matrix(val_data[target_column], predictions), annot=True, fmt="d")  # Annotate and format
      plt.show()  # Display the heatmap
    except ModuleNotFoundError:
      print("\n** seaborn library not found for visualization. Confusion matrix values:")
      print(confusion_matrix(val_data[target_column], predictions))
    except Exception as e:
      print(f"\n** Error occurred while visualizing the confusion matrix: {e}")
    else:
      print("\n** Confusion matrix visualized (using seaborn if available).**")

  else:
    print("\n** Skipping confusion matrix visualization.**")


########################################################################################
# Learning Curves
########################################################################################   
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def plot_learning_curves(trained_model, train_data, val_data, target_column):
  """
  Guides the user through plotting learning curves for the trained model.

  Args:
      trained_model (object): The trained classification or regression model.
      train_data (pandas.DataFrame): The DataFrame containing the training data (features and target).
      val_data (pandas.DataFrame): The DataFrame containing the validation data (features and target).
      target_column (str): The name of the target column.
  """
  print("Learning curves show how the model's performance changes with the amount of training data.")
  print("\n** Would you like to visualize learning curves? (y/n) **")
  
  while True:
    choice = input().lower()
    if choice in ["y", "n"]:
      break
    else:
      print("Invalid choice. Please choose 'y' or 'n'.")

  if choice == "y":
    print("\n** Visualizing learning curves...**")
    
    # Extract features and target
    X = train_data.drop(target_column, axis=1)
    y = train_data[target_column]

    # Define train sizes
    train_sizes = np.linspace(0.1, 1.0, 10)

    # Choose appropriate scoring metric
    if isinstance(trained_model, (LogisticRegression, RandomForestClassifier)):
      scoring = "accuracy"
    else:  # Assuming it's a regression model
      scoring = "neg_mean_squared_error"

    # Compute learning curve
    try:
      train_sizes, train_scores, val_scores = learning_curve(
        trained_model, X, y, 
        train_sizes=train_sizes, 
        cv=5, 
        scoring=scoring,
        n_jobs=-1,
        verbose=0
      )
    except Exception as e:
      print(f"An error occurred while computing the learning curve: {str(e)}")
      print("Unable to plot learning curves.")
      return

    # Compute mean and standard deviation
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)

    # Plot learning curve
    print("Close the plot window to continue with this program.")
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, val_mean, 'o-', color="g", label="Cross-validation score")
    
    # Plot standard deviation bands
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color="g")
    
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.title("Learning Curves")
    plt.legend(loc="best")
    plt.grid(True)
    
    plt.show()
  else:
    print("\n** Skipping learning curves visualization.**")

########################################################################################
# ROC Curves (for Classification)
########################################################################################
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from itertools import cycle

def plot_roc_curve(trained_model, val_data, target_column):
  """
  Guides the user through plotting ROC curves for the trained classification model.

  Args:
      trained_model (object): The trained classification model.
      val_data (pandas.DataFrame): The DataFrame containing the validation data (features and target).
      target_column (str): The name of the target column.
  """

  # Check if the model is a classification model
  if not hasattr(trained_model, "predict_proba"):
    print("This model doesn't support probability predictions. ROC curve cannot be plotted.")
    return

  print("ROC curves show the trade-off between true positive rates (TPR) and false positive rates (FPR) at different classification thresholds.")

  print("\n** Would you like to visualize ROC curves? (y/n) **")
  while True:
    choice = input().lower()
    if choice in ["y", "n"]:
      break
    else:
      print("Invalid choice. Please choose 'y' or 'n'.")

  if choice == "y":
    print("\n** Visualizing ROC curves...**")

    # Extract features and target
    X_val = val_data.drop(target_column, axis=1)
    y_val = val_data[target_column]

    # Get the number of classes
    n_classes = len(np.unique(y_val))

    # Binarize the output for multiclass
    y_val_bin = label_binarize(y_val, classes=np.unique(y_val))

    # Predict probabilities
    y_pred_proba = trained_model.predict_proba(X_val)

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
      fpr[i], tpr[i], _ = roc_curve(y_val_bin[:, i], y_pred_proba[:, i])
      roc_auc[i] = auc(fpr[i], tpr[i])

    print("Close the plot window to continue with this program.")

    # Plot ROC curves
    plt.figure(figsize=(10, 8))
    colors = cycle(['blue', 'red', 'green', 'yellow', 'purple', 'orange'])
    
    for i, color in zip(range(n_classes), colors):
      plt.plot(fpr[i], tpr[i], color=color, lw=2,
               label=f'ROC curve of class {i} (area = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()

  else:
    print("\n** Skipping ROC curve visualization.**")


########################################################################################
# Precision-Recall Curves (for Classification)
########################################################################################
def plot_precision_recall_curve(trained_model, val_data, target_column):
    if not hasattr(trained_model, "predict_proba"):
        print("\n** This function is only applicable for classification models with probability prediction capabilities. Skipping precision-recall curve.**")
        return

    print("Precision-recall curves show the trade-off between precision and recall at different classification thresholds.")
    if not user_confirms("Would you like to visualize precision-recall curves?"):
        return

    X_val = val_data.drop(target_column, axis=1)
    y_val = val_data[target_column]
    
    # Handle multiclass
    n_classes = len(np.unique(y_val))
    if n_classes > 2:
        y_val_bin = label_binarize(y_val, classes=np.unique(y_val))
        y_pred_proba = trained_model.predict_proba(X_val)
        
        plt.figure(figsize=(10, 8))
        for i in range(n_classes):
            precision, recall, _ = precision_recall_curve(y_val_bin[:, i], y_pred_proba[:, i])
            plt.plot(recall, precision, label=f'Class {i}')
    else:
        y_pred_proba = trained_model.predict_proba(X_val)[:, 1]
        precision, recall, _ = precision_recall_curve(y_val, y_pred_proba)
        plt.plot(recall, precision)

    print("Close the plot window to continue with this program.")

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    if n_classes > 2:
        plt.legend()
    plt.grid(True)
    plt.show()

########################################################################################
# SHAP Explanations
########################################################################################
def explain_with_shap(trained_model, val_data, target_column, explainer_type="force_plot"):
    if not SHAP_AVAILABLE:
        print("\n** SHAP library not found. SHAP explanations are unavailable.**")
        return

    if not user_confirms("Would you like to generate SHAP explanations?"):
        return

    print("\n** Generating SHAP explanations (may take some time)...**")

    X_val = val_data.drop(target_column, axis=1)
    instance = X_val.iloc[0].values.reshape(1, -1)

    if explainer_type == "tree_explainer" and hasattr(trained_model, "tree_"):
        explainer = shap.TreeExplainer(trained_model)
    else:
        explainer = shap.Explainer(trained_model.predict, instance)

    shap_values = explainer(instance)

    try:
        print("Close the plot window to continue with this program.")
        shap.plots.waterfall(shap_values[0])
    except:
        print("\n** Unable to display SHAP explanation. Try a different explainer type.")

########################################################################################
# Partial Dependence Plots (PDPs)
########################################################################################
def plot_partial_dependence(trained_model, val_data, target_column, feature_names=None):
    if not user_confirms("Would you like to visualize Partial Dependence Plots (PDPs)?"):
        return

    X_val = val_data.drop(target_column, axis=1)
    feature_names = feature_names or X_val.columns.tolist()

    while True:
        print("\n** Select a feature to visualize its PDP: ")
        for i, feature_name in enumerate(feature_names, 1):
            print(f"{i}. {feature_name}")
        print(f"{len(feature_names) + 1}. Exit")

        feature_choice = get_valid_input("Enter your choice: ", lambda x: 1 <= int(x) <= len(feature_names) + 1)
        
        if feature_choice == len(feature_names) + 1:
            break

        feature_index = feature_choice - 1
        
        pdp = partial_dependence(trained_model, X_val, features=[feature_names[feature_index]])
        print("Close the plot window to continue with this program.")

        plt.figure(figsize=(10, 6))
        plt.plot(pdp['values'][0], pdp['average'][0])
        plt.xlabel(feature_names[feature_index])
        plt.ylabel("Partial Dependence")
        plt.title(f"Partial Dependence Plot for {feature_names[feature_index]}")
        plt.grid(True)
        plt.show()

        if not user_confirms("Would you like to visualize another PDP?"):
            break


########################################################################################
# Feature Importance Analysis
########################################################################################
def analyze_feature_importance(trained_model, val_data, target_column, feature_names=None):
    if not user_confirms("Would you like to analyze feature importance?"):
        return

    X_val = val_data.drop(target_column, axis=1)
    y_val = val_data[target_column]
    feature_names = feature_names or X_val.columns.tolist()

    results = permutation_importance(trained_model, X_val, y_val, n_repeats=10, random_state=42)
    
    feature_importances = sorted(zip(feature_names, results.importances_mean), key=lambda x: x[1], reverse=True)
    print("Close the plot window to continue with this program.")

    plt.figure(figsize=(10, 6))
    plt.barh([f[0] for f in feature_importances], [f[1] for f in feature_importances])
    plt.xlabel("Importance")
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.show()

########################################################################################
# Model Persistence
########################################################################################
def save_model(trained_model, save_path):
    if not user_confirms("Would you like to save the trained model?"):
        return

    try:
        with open(save_path, 'wb') as f:
            pickle.dump(trained_model, f)
        print(f"\n** Model saved successfully to: {save_path} **")
    except Exception as e:
        print(f"\n** Error saving the model: {e} **")

########################################################################################
# Helper Functions
########################################################################################
def user_confirms(message):
    while True:
        choice = input(f"{message} (y/n): ").lower()
        if choice in ["y", "n"]:
            return choice == "y"
        print("Invalid choice. Please choose 'y' or 'n'.")

def get_valid_input(prompt, validator):
    while True:
        try:
            user_input = int(input(prompt))
            if validator(user_input):
                return user_input
        except ValueError:
            pass
        print("Invalid input. Please try again.")