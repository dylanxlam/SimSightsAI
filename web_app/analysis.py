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

  print("\nWe will start by generating a classification report. This report provides a detailed breakdown of the")
  print("model's performance for each class in the classification task.")
  print("Generating a classification report...")




  # Generate and display the report
  report = classification_report(val_data[target_column], predictions, output_dict=True)
  print("\nClassification Report:")
  print("  * To know which class number corresponds to which class of your feature, scroll up to observe the class distribution")
  print("    printed after selecting your target column.")
  print("  * The columns printed will both be in the same order. Class: 0 will be first on the class distribution print, Class: 1, will be second, and so forth.")
  for class_name, metrics in report.items():
    if isinstance(metrics, dict):
      print(f"\nClass: {class_name}")
      for metric_name, value in metrics.items(): 
        print(f"  - {metric_name}: {value:.4f}")  # Format metric values
    else:
      # Handle the 'accuracy' metric separately
      print(f"\n\nOverall Accuracy: {metrics:.4f}")

  print("""
  Interpreting Your Classification Report:

  This report summarizes the performance of your classification model on different classes. Here's a breakdown of key metrics:

    * Support: The total number of true instances for a particular class in the dataset.
    * Precision: Proportion of predicted positives that are actually positive (out of all predicted positives).
    * Recall: Proportion of actual positives that are correctly identified (out of all actual positives).
    * F1-Score: Harmonic mean of precision and recall, combining both metrics into a single score (generally 
        preferred over accuracy for imbalanced datasets).
    * Accuracy: Proportion of total predictions that were correct (be cautious when interpreting accuracy in imbalanced datasets).

  Interpretations:

    * High precision for a class indicates the model is good at identifying that class without many false positives.
    * High recall for a class indicates the model is good at finding most of the actual positives for that class.
    * A balanced F1-score suggests a good balance between precision and recall.
    * A high overall accuracy might be misleading in imbalanced datasets where the model might simply predict the majority class most of the time.

  Additional Tips:

    * Consider using visualization tools to understand per-class performance.
    * Analyze the confusion matrix to identify potential issues like class imbalance or misclassifications.
    * Depending on your task, prioritize metrics that best reflect your desired outcome (e.g., high recall for fraud detection).

  This report provides insights into your model's strengths and weaknesses for each class. Use it to guide 
    further improvements and optimize your model for your specific use case.
  """)

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
  print("\n\n\nNext, we can create a confusion matrix.")
  print("Confusion matrices help us understand how often the model correctly classified each class and where it made mistakes.")
  print("Would you like to visualize the confusion matrix? (y/n)")

  while True:
    choice = input().lower()
    if choice in ["y", "n"]:
      break
    else:
      print("Invalid choice. Please choose 'y' or 'n'.")

  if choice == "y":
    # Generate and display the confusion matrix (using library like seaborn)
    print("\nVisualizing the confusion matrix...")

    try:
      import seaborn as sns  # Import seaborn for visualization (optional)
      import matplotlib.pyplot as plt  # Explicitly import matplotlib for plt.show()
      print("""
      Here's how to interpret the confusion matrix:

        * True Positives (TP): These are the correct predictions of the positive class.
        * True Negatives (TN): These are the correct predictions of the negative class.
        * False Positives (FP): These are the incorrect predictions of the positive class (also known as Type I errors).
        * False Negatives (FN): These are the incorrect predictions of the negative class (also known as Type II errors).

      Interpretations:

        * High accuracy (diagonal elements) and low off-diagonal elements generally indicate a good model.
        * High False Positives (FP) suggest the model is over-predicting the positive class.
        * High False Negatives (FN) suggest the model is under-predicting the positive class.

      Additional Tips:

        * Consider using metrics like precision, recall, and F1-score for a more comprehensive evaluation.
        * Analyze the confusion matrix for specific classes if your data is imbalanced.
      """)

      print("\nClose the plot window to continue with this program.")
      sns.heatmap(confusion_matrix(val_data[target_column], predictions), annot=True, fmt="d")  # Annotate and format
      plt.show()  # Display the heatmap
    except ModuleNotFoundError:
      print("\nseaborn library not found for visualization. Confusion matrix values:")
      print(confusion_matrix(val_data[target_column], predictions))
    except Exception as e:
      print(f"\nError occurred while visualizing the confusion matrix: {e}")
    else:
      print("")

  else:
    print("\nSkipping confusion matrix visualization.")


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
  print("\n\n\nNext, we can visualize learning curves.")
  print("Learning curves show how the model's performance changes with the amount of training data.")
  print("Would you like to visualize learning curves? (y/n)")
  
  while True:
    choice = input().lower()
    if choice in ["y", "n"]:
      break
    else:
      print("Invalid choice. Please choose 'y' or 'n'.")

  if choice == "y":
    print("\nVisualizing learning curves...")
    
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
    print("""
    Here's how to interpret Learning Curves:

    * The x-axis represents the size of the training set used to train the model.
    * The y-axis represents the model's performance metric (e.g., accuracy, loss).
    * Two curves are typically plotted: training score and validation score.

    Interpretations:

      * A decreasing training score indicates the model is learning the training data patterns.
      * A stable or slightly increasing validation score indicates the model is generalizing well to unseen data (avoiding overfitting).
      * A large gap between training and validation scores suggests potential overfitting. 
          - The model is memorizing training data patterns that don't generalize well.

    Tips:

      * Look for a smooth curve for both training and validation scores.
      * If the validation score starts decreasing after a certain point, it might indicate underfitting.
          - The model might not be learning enough from the data.

    Additional Considerations:

      * The ideal learning curve shape can vary depending on the model and data complexity.
      * Techniques like regularization can help reduce overfitting and improve generalization.
    """)

    print("\nClose the plot window to continue with this program.")
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
    print("\nSkipping learning curves visualization.")

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

  print("\n\n\nNext, we can visualize ROC curves.")
  print("ROC curves show the trade-off between true positive rates (TPR) and false positive rates (FPR) at different classification thresholds.")
  print("Would you like to visualize ROC curves? (y/n) ")
  while True:
    choice = input().lower()
    if choice in ["y", "n"]:
      break
    else:
      print("Invalid choice. Please choose 'y' or 'n'.")

  if choice == "y":
    print("\nVisualizing ROC curves...")

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


    print("""
    Here's how to interpret ROC Curves:

    * The x-axis represents the False Positive Rate (FPR), the proportion of negative instances incorrectly classified as positive.
    * The y-axis represents the True Positive Rate (TPR), the proportion of positive instances correctly classified as positive.
    * An ideal ROC curve approaches the top-left corner, indicating the model can distinguish positive and negative classes very well.

    Interpretations:

      * A curve closer to the top-left corner indicates better model performance in distinguishing classes.
      * A curve that follows the diagonal line (FPR = TPR) indicates random guessing.
      * A curve that dips below the diagonal suggests the model performs worse than random guessing.

    Additional Considerations:

      * The Area Under the ROC Curve (AUC) summarizes the overall performance. A higher AUC indicates better class discrimination.
      * ROC curves are useful for comparing the performance of different models on imbalanced datasets.

    Tips:

      * Use ROC curves in conjunction with other evaluation metrics for a comprehensive picture.
      * Consider using techniques like cost-sensitive learning if dealing with imbalanced classes where misclassifications have different costs.
    """)

    print("\nClose the plot window to continue with this program.")

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
    print("\nSkipping ROC curve visualization.")


########################################################################################
# Precision-Recall Curves (for Classification)
########################################################################################
def plot_precision_recall_curve(trained_model, val_data, target_column):
    if not hasattr(trained_model, "predict_proba"):
        print("\nThis function is only applicable for classification models with probability prediction capabilities. Skipping precision-recall curve.")
        return
    
    print("\n\n\nNext, we can visualize precision-recall curves.")
    print("Precision-recall curves show the trade-off between precision and recall at different classification thresholds.")
    if not user_confirms("Would you like to visualize precision-recall curves?"):
        print("\nSkipping precision-recall curves visualization.")
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


    print("""
    Here's how to interpret Precision-Recall Curves:

    * The x-axis represents the Recall, the proportion of positive instances correctly identified.
    * The y-axis represents the Precision, the proportion of predicted positive instances that are truly positive.
    * A good precision-recall curve strives for a balance between high recall (catching most positive cases) 
        and high precision (minimizing false positives).

    Interpretations:

      * A curve closer to the top-right corner indicates a good balance between precision and recall.
      * A curve that leans heavily towards the left (high recall, low precision) suggests the model predicts many positives but with a 
          high rate of false positives.
      * A curve that leans heavily towards the bottom (low recall, high precision) suggests the model is cautious and misses some 
          true positives to avoid false positives.

    Additional Considerations:

      * The Area Under the Precision-Recall Curve (AUPRC) summarizes the overall performance. A higher AUPRC indicates a better balance 
          between precision and recall.
      * Precision-recall curves are particularly useful for imbalanced datasets where the cost of misclassification might be unequal.

    Tips:

      * Use precision-recall curves in conjunction with other metrics like F1-score for a more comprehensive evaluation.
      * Consider adjusting model thresholds to trade off between precision and recall depending on your specific needs.

    """)


    print("\nClose the plot window to continue with this program.")

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
        print("\nSHAP library not found. SHAP explanations are unavailable.")
        return

    print("\n\n\nNext, we can visualize SHAP explanations.")
    print("SHAP (SHapley Additive exPlanations) is a technique for explaining the predictions of any machine learning model.")
    print("It uses game theory concepts to assign a fair share of credit (SHAP value) to each feature for the model's prediction.")
    if not user_confirms("Would you like to generate SHAP explanations?"):
        print("\nSkipping SHAP explanations.")
        return

    print("\nGenerating SHAP explanations (may take some time)...")

    X_val = val_data.drop(target_column, axis=1)
    instance = X_val.iloc[0].values.reshape(1, -1)

    if explainer_type == "tree_explainer" and hasattr(trained_model, "tree_"):
        explainer = shap.TreeExplainer(trained_model)
    else:
        explainer = shap.Explainer(trained_model.predict, instance)

    shap_values = explainer(instance)

    try:
        print("""
        Here's how to interpret SHAP Explanations:

        SHAP (SHapley Additive exPlanations) is a technique for explaining the predictions of any machine learning model. It uses 
          game theory concepts to assign a fair share of credit (SHAP value) to each feature for the model's prediction.

        Interpretations:

          * A positive SHAP value for a feature indicates it increased the model's prediction.
          * A negative SHAP value indicates it decreased the model's prediction.
          * The magnitude of the SHAP value reflects the strength of the feature's influence.
          * SHAP summary plots visualize the distribution of SHAP values for each feature, helping identify important features.
          * SHAP force plots for individual instances show how each feature contributes to a specific prediction.

        Benefits:

          * SHAP explanations are model-agnostic and work with various machine learning models.
          * They provide local explanations, explaining specific predictions rather than just global feature importance.
          * SHAP values are additive, allowing for easier interpretation of feature interactions.

        Additional Considerations:

          * SHAP explanations are computationally expensive for complex models and large datasets.

        Tips:

          * Use SHAP summary plots to identify the most influential features for your model.
          * Analyze SHAP force plots for individual instances to understand how features contribute to specific predictions.
        """)


        print("\nClose the plot window to continue with this program.")
        shap.plots.waterfall(shap_values[0])
    except:
        print("\nUnable to display SHAP explanation. Try a different explainer type.")

########################################################################################
# Partial Dependence Plots (PDPs)
########################################################################################
def plot_partial_dependence(trained_model, val_data, target_column, feature_names=None):
    print("\n\n\nNext, we can visualize partial dependence plots (PDPs).")
    print("PDPs visualize the average effect of a single feature on the model's predictions while averaging out the effects of other features.")
    if not user_confirms("Would you like to visualize Partial Dependence Plots (PDPs)?"):
        print("\nSkipping partial dependence plot visualization.")
        return

    X_val = val_data.drop(target_column, axis=1)
    feature_names = feature_names or X_val.columns.tolist()

    while True:
        print("\nSelect a feature to visualize its PDP: ")
        for i, feature_name in enumerate(feature_names, 1):
            print(f"{i}. {feature_name}")
        print(f"{len(feature_names) + 1}. Exit")

        feature_choice = get_valid_input("Enter your choice: ", lambda x: 1 <= int(x) <= len(feature_names) + 1)
        
        if feature_choice == len(feature_names) + 1:
            break

        feature_index = feature_choice - 1
        
        pdp = partial_dependence(trained_model, X_val, features=[feature_names[feature_index]])
        print("""
        Here's how to interpret Partial Dependence Plots (PDPs):

        * PDPs visualize the average effect of a single feature on the model's predictions while averaging out the effects of other features.
        * The x-axis represents the values of the feature of interest.
        * The y-axis represents the average predicted value of the target variable.

        Interpretations:

          * An increasing or decreasing trend in the PDP indicates a positive or negative relationship between the feature and the target variable.
          * A flat line suggests the feature has little to no impact on the prediction.
          * Non-monotonic relationships (up-and-down patterns) indicate complex interactions with other features.

        Additional Considerations:

          * PDPs are useful for understanding feature importance and identifying potential interactions between features.
          * They don't necessarily represent causal relationships.

        Tips:

          * Consider using PDPs in conjunction with other feature importance techniques for a more holistic view.
          * Explore PDPs for different features to gain insights into your model's behavior.
        """)


        print("\nClose the plot window to continue with this program.")

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
    print("\n\n\nNext, we can visualize feature importances.")
    print("Feature importance plots visualize the relative importance of features in a machine learning model for predicting the target variable. ")
    if not user_confirms("Would you like to analyze feature importance?"):
        return

    X_val = val_data.drop(target_column, axis=1)
    y_val = val_data[target_column]
    feature_names = feature_names or X_val.columns.tolist()

    results = permutation_importance(trained_model, X_val, y_val, n_repeats=10, random_state=42)
    
    feature_importances = sorted(zip(feature_names, results.importances_mean), key=lambda x: x[1], reverse=True)
    print("""
    Understanding Feature Importance Plots:

    Feature importance plots visualize the relative importance of features in a machine learning model for predicting the target variable. 
    These plots help identify which features contribute most to the model's predictions.

    Common Visualization Types:

      * Bar charts: Features are represented by bars, with bar height indicating importance.
      * Feature weights: Features are listed with their corresponding importance scores.

    Interpretations:

      * Features with higher importance scores have a greater influence on the model's predictions.
      * These plots help prioritize features for further analysis or model optimization.

    Limitations:

      * Importance scores can vary depending on the feature importance technique used (e.g., permutation importance, tree-based feature importance).
      * Feature importance doesn't necessarily translate directly to feature causality or interpretability.

    Tips:

      * Use feature importance plots in conjunction with other techniques like SHAP explanations for a deeper understanding of feature influence.
      * Consider exploring different feature importance techniques to see if the results are consistent.

    """)

    print("\nClose the plot window to continue with this program.")

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
    print("""
    Saving a model to a pickle file (.pkl) offers several advantages:

      * Persistence: Saves the trained model's state, including learned parameters (weights and biases) and structure of the model, 
        allowing you to load and reuse it later without re-training.

      * Faster Predictions: Once loaded, the model can make predictions on new data 
        much faster than re-training it for each prediction.

      * Efficient Evaluation: Enables you to evaluate the model's performance on 
        different datasets without re-training each time.

      * Deployment Potential: Saved models can be deployed to web applications 
        or production environments for real-time predictions.

      * Collaboration and Sharing: Allows you to share the model with others who can 
        use it for predictions or further analysis.

    In essence, saving a model is like capturing its learned knowledge, 
    making it readily available for future use and collaboration.
    """)

    if not user_confirms("\nWould you like to save the trained model?"):
        return

    try:
        with open(save_path, 'wb') as f:
            pickle.dump(trained_model, f)
        print(f"\nModel saved successfully to: {save_path}")
        print("\nHere's an example of how to use the pkl file to predict on new data:")
        print("\n```python")
        print("import pickle")
        print("\n# Replace 'model.pkl' with the actual filename of your pickle file")
        print("with open('model.pkl', 'rb') as file:")
        print("  loaded_model = pickle.load(file)")
        print("\n# Assuming your new data is stored in a variable called 'new_data'")
        print("predictions = loaded_model.predict(new_data)")
        print("```")
    except Exception as e:
        print(f"\nError saving the model: {e}")

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