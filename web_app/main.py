from cleaning import read_data, handle_missing_values, identify_and_handle_outliers, handle_duplicates, handle_formatting
from eda import visualize_numerical, visualize_categorical, analyze_correlations
from preprocessing import convert_data_types, scale, create_interaction_features, create_feature_bins, create_custom_features, create_one_hot_encoding, create_label_encoding, handle_class_imbalance, feature_selection
from modeling import model_selection, data_splitting, train_model, tune_hyperparameters, evaluate_model
from analysis import generate_classification_report, visualize_confusion_matrix, plot_learning_curves, plot_roc_curve, plot_precision_recall_curve, explain_with_shap, plot_partial_dependence, analyze_feature_importance, save_model

def main():
    # Reading Data
    # Get user input for file path
    filepath = input("\n\nPlease enter the path to your data file (CSV, Excel, TSV, JSON): ")

    try:
        # Read data using the function
        data = read_data(filepath)
        # Process or analyze the data here (assuming data is a DataFrame)
        print("Data loaded successfully!")
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        data = read_data(filepath)

    # Cleaning Data
    cleaned_data = handle_missing_values(data.copy())
    identify_and_handle_outliers(cleaned_data)
    handle_duplicates(cleaned_data)
    handle_formatting(cleaned_data)

    # Exploratory Data Analysis
    data_descriptive_statistics = data.describe()
    print("Descriptive Statistics of Your Data:\n", data_descriptive_statistics)
    visualize_numerical(cleaned_data)
    visualize_categorical(cleaned_data)
    analyze_correlations(cleaned_data)

    # Preprocess Data
    data_types = data.dtypes
    print("Data Types for Each Column in Your Data", data_types)
    preprocessed_data = convert_data_types(cleaned_data)
    scale(preprocessed_data)
    create_interaction_features(preprocessed_data, categorical_cols=None)
    create_feature_bins(preprocessed_data, continuous_cols=None, n_bins=5)
    create_custom_features(preprocessed_data)
    create_one_hot_encoding(preprocessed_data, categorical_cols=None)
    create_label_encoding(preprocessed_data, categorical_cols=None)
    target_column = input("Enter the name of the column containing the target variable (the variable you wish to predict/classify):")
    handle_class_imbalance(preprocessed_data, target_column)
    feature_selection(preprocessed_data, target_column)

    # Modeling
    chosen_model = model_selection(preprocessed_data, target_column)
    train_data, val_data, test_data = data_splitting(preprocessed_data, target_column, test_size=0.2, random_state=42)
    trained_model = train_model(chosen_model, train_data, val_data, test_data, target_column, hyperparameter_options=None)
    hyperparameter_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10],
    'solver': ['liblinear', 'lbfgs', 'sag']  # Example hyperparameters for LogisticRegression
    }
    model_class = chosen_model.__class__ 
    best_model = tune_hyperparameters(model_class, train_data, val_data, test_data, target_column, hyperparameter_grid)
    trained_model = train_model(best_model, train_data, val_data, test_data, target_column)

    evaluate_model(trained_model, train_data, val_data, test_data, target_column)
    metric_value, predictions = evaluate_model(trained_model, train_data, val_data, test_data, target_column)

    # Model Analysis
    generate_classification_report(trained_model, val_data, predictions, target_column)
    visualize_confusion_matrix(trained_model, val_data)
    plot_learning_curves(trained_model, train_data, val_data)
    plot_roc_curve(trained_model, val_data)
    plot_precision_recall_curve(trained_model, val_data)
    explain_with_shap(trained_model, val_data, explainer_type="force_plot")
    plot_partial_dependence(trained_model, val_data, feature_names=None)
    analyze_feature_importance(trained_model, val_data, feature_names=None)
    save_path = input("Enter the path (including filename) to save the model:")
    save_model(trained_model, save_path)

if __name__ == "__main__":
    main()