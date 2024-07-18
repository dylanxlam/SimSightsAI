from cleaning import read_data, handle_missing_values, identify_and_handle_outliers, handle_duplicates, handle_formatting
from eda import visualize_numerical, visualize_categorical, analyze_correlations
from preprocessing import convert_data_types, scale, create_interaction_features, create_feature_bins, create_custom_features, create_one_hot_encoding, create_label_encoding, handle_class_imbalance, feature_selection
from modeling import model_selection, data_splitting, train_model, tune_hyperparameters, evaluate_model
from analysis import generate_classification_report, visualize_confusion_matrix, plot_learning_curves, plot_roc_curve, plot_precision_recall_curve, explain_with_shap, plot_partial_dependence, analyze_feature_importance, save_model

def main():
    # Reading Data
    # Get user input for file path
    print("\n\nWelcome to SimSightsAI! Here we will walk you through the machine learning pipeline to find valuable insights and create\npredictive models with your data.")
    filepath = input("\n\nPlease enter the path to your data file (CSV, Excel, TSV, JSON): ")

    try:
        # Read data using the function
        data = read_data(filepath)
        # Process or analyze the data here (assuming data is a DataFrame)
        print("\nData loaded successfully!")
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        data = read_data(filepath)

    # Cleaning Data
    print("\n\n\n\n\n\nWe will start off by cleaning your data. It is very important to make sure your data is clean of inconsistencies and\nformatting errors before you begin to work with it.")
    print("In this section we will work to identify and handle potential missing values, outliers, duplicates, and formatting inconsistencies!\n\n")
    cleaned_data = handle_missing_values(data.copy())
    identify_and_handle_outliers(cleaned_data)
    handle_duplicates(cleaned_data)
    handle_formatting(cleaned_data)
    print("\nYour potential missing values, outliers, duplicates, and formatting inconsistencies are now handled!")

    # Exploratory Data Analysis
    print("\n\n\n\n\n\nNow that your data is clean, we can move on to explore your data to create initial visualizations and learn about the behavior of your data.")
    data_descriptive_statistics = cleaned_data.describe()
    print("\nWe will start by displaying the descriptive statistics of your cleaned data:\n", data_descriptive_statistics)
    visualize_numerical(cleaned_data)
    visualize_categorical(cleaned_data)
    analyze_correlations(cleaned_data)

    # Preprocess Data
    print("\n\n\n\n\n\nNow that we have explored and learned about your data's behavior, we can move on to preprocess the data for modeling!")
    data_types = data.dtypes
    print("\nData Types for Each Column in Your Data\n", data_types)
    preprocessed_data = convert_data_types(cleaned_data)
    scale(preprocessed_data)
    create_interaction_features(preprocessed_data, categorical_cols=None)
    create_feature_bins(preprocessed_data, continuous_cols=None, n_bins=5)
    create_custom_features(preprocessed_data)
    create_one_hot_encoding(preprocessed_data, categorical_cols=None)
    create_label_encoding(preprocessed_data, categorical_cols=None)
    print("")
    print("Next, enter the name of the column containing the target variable (the variable you wish to predict/classify).")
    print("Here are the columns to choose from:\n")
    for column in preprocessed_data.columns:
        print("  - ", column)
    target_column = input("Enter the variable you want to predict:")
    handle_class_imbalance(preprocessed_data, target_column)
    feature_selection(preprocessed_data, target_column)

    # Modeling
    print("\n\n\n\n\n\nNow that we have preprocessed your data for modeling, we can move on creating your predictive model!")
    train_data, val_data, test_data = data_splitting(preprocessed_data, target_column)
    chosen_model, train_data, val_data, test_data = model_selection(train_data, val_data, test_data, target_column)

    model_class = chosen_model.__class__ 
    if model_class.__name__ == 'LogisticRegression':
        hyperparameter_grid = {
            'C': [0.001, 0.01, 0.1, 1, 10],  # Regularization parameter
            'solver': ['liblinear', 'lbfgs', 'sag'],  # Optimization algorithm
            'max_iter': [2000]  # Maximum number of iterations
        }
    elif model_class.__name__ == 'LinearRegression':
        hyperparameter_grid = {
            'fit_intercept': [True, False],  # Whether to fit an intercept term
            'positive': [True, False],  # Constraint the coefficients to be positive
        }
    elif model_class.__name__ == 'RandomForestClassifier' or model_class.__name__ == 'RandomForestRegressor':
        hyperparameter_grid = {
            'n_estimators': [100, 200, 500],  # Number of trees in the forest
            'max_depth': [4, 8, 16],  # Maximum depth of individual trees
            'min_samples_split': [2, 5, 10],  # Minimum samples required to split a node
            'min_samples_leaf': [1, 2, 4],  # Minimum samples required at each leaf node
        }
    else:
        # Handle other model classes (you can add more elif blocks for other models)
        print(f"Hyperparameter tuning not currently supported for {model_class.__name__}")
        hyperparameter_grid = {}

        
    best_model, train_data, val_data, test_data = tune_hyperparameters(model_class, train_data, val_data, test_data, target_column, hyperparameter_grid)

    trained_model = train_model(best_model, train_data, val_data, test_data, target_column)

    metric_value, predictions = evaluate_model(trained_model, train_data, val_data, test_data, target_column)

    # Model Analysis
    print("\n\n\n\n\n\nNow that we have created a machine learning model for your data, we can move onto analyzing your model and its performance!")
    from sklearn.base import is_classifier
    if is_classifier(model_class):
        generate_classification_report(trained_model, val_data, predictions, target_column)
        visualize_confusion_matrix(trained_model, val_data, predictions, target_column)
        plot_roc_curve(trained_model, val_data, target_column)
        plot_precision_recall_curve(trained_model, val_data, target_column)
    plot_learning_curves(trained_model, train_data, val_data, target_column)

    explain_with_shap(trained_model, val_data, target_column, explainer_type="force_plot")
    plot_partial_dependence(trained_model, val_data, target_column, feature_names=None)
    analyze_feature_importance(trained_model, val_data, target_column, feature_names=None)
    print("\n\n\nNow that we have analyzed your model and its performance, we can now save your model into a pkl file.")
    print("Enter the path (including filename) to save the model: ")
    print("\nFor example:")
    print("* On Windows: C:\\Users\\Username\\Documents\\my_model.pkl")
    print("* On macOS/Linux: /Users/Username/Documents/my_model.pkl")
    save_path = input()

    save_model(trained_model, save_path)

    print("\n\nThis wraps up our program. Thank you for using SimSightsAI. Rerun this program anytime to revise your model, or create a new one!")

if __name__ == "__main__":
    main()