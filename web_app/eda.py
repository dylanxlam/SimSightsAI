########################################################################################
# Import Statements
########################################################################################
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
import numpy as np


########################################################################################
# Descriptive Statistics
########################################################################################

########################################################################################
# Data Distribution Visualization
########################################################################################
def visualize_numerical(data):
  """
  Prompts the user to choose a numeric feature and a visualization type.

  Args:
      data (pandas.DataFrame): The DataFrame containing the data.
  """
  numeric_cols = [col for col in data.columns if pd.api.types.is_numeric_dtype(data[col])]
  if not numeric_cols:
    print("No numeric features found in the data.")
    return

  print("Choose a numeric feature to visualize:")
  for i, col in enumerate(numeric_cols):
    print(f"  {i+1}. {col}")

  while True:
    try:
      feature_choice = int(input("Enter your choice (1-{} or 'q' to quit): ".format(len(numeric_cols))))
      if 1 <= feature_choice <= len(numeric_cols):
        col = numeric_cols[feature_choice - 1]
        print("Choose a visualization type:")
        print("  1. Histogram (shows the distribution of the data)")
        print("  2. Box Plot (shows quartiles and potential outliers)")
        print("  3. Scatter Plot (choose another numeric feature to explore relationships)")
        print("  4. Violin Plot (shows the distribution of the data with a hint of skewness)")
        print("  5. Kernel Density Plot (shows the probability density of the data)")
        print("  'q' to quit")

        viz_choice = input("Enter your choice (1-5 or 'q'): ")
        if viz_choice.lower() == 'q':
          print("Exiting visualization.")
          return
        else:
          try:
            viz_choice = int(viz_choice)
            if 1 <= viz_choice <= 5:
              if viz_choice == 1:
                data[col].hist()
                plt.xlabel(col)
                plt.ylabel("Number of observations")
                plt.title(f"Distribution of {col}")
                plt.show()
              elif viz_choice == 2:
                data.boxplot(column=col)
                plt.xlabel(col)
                plt.ylabel("Value")
                plt.title(f"Box Plot of {col}")
                plt.show()
              elif viz_choice == 3:
                other_col = choose_another_numeric_feature(data, col)
                if other_col:
                  data.plot.scatter(x=col, y=other_col)
                  plt.xlabel(col)
                  plt.ylabel(other_col)
                  plt.title(f"Scatter Plot of {col} vs {other_col}")
                  plt.show()
              elif viz_choice == 4:
                data[col].plot(kind="violin")
                plt.xlabel(col)
                plt.ylabel("Density")
                plt.title(f"Violin Plot of {col}")
                plt.show()
              elif viz_choice == 5:
                data[col].plot(kind="density")
                plt.xlabel(col)
                plt.ylabel("Density")
                plt.title(f"Kernel Density Plot of {col}")
                plt.show()
              else:
                print("Invalid visualization choice.")
            else:
              print("Invalid visualization choice. Please choose between 1 and 5.")
          except ValueError:
            print("Invalid input. Please enter a number.")
      else:
        print("Invalid choice. Please enter a number between 1 and {} or 'q' to quit.".format(len(numeric_cols)))
    except:
      pass  # Placeholder to avoid needing a specific exception type


def choose_another_numeric_feature(data, current_col):
  """
  Prompts the user to choose another numeric feature for a scatter plot.

  Args:
      data (pandas.DataFrame): The DataFrame containing the data.
      current_col (str): The name of the already chosen numeric feature.

  Returns:
      str: The name of the chosen numeric feature (excluding the current one), 
          or None if the user cancels.
  """
  remaining_numeric_cols = [col for col in data.columns 
                             if pd.api.types.is_numeric_dtype(data[col]) and col != current_col]
  if not remaining_numeric_cols:
    print(f"No other numeric features available besides {current_col}.")
    return None

  print(f"Choose another numeric feature to create a scatter plot with {current_col}:")
  for i, col in enumerate(remaining_numeric_cols):
    print(f"  {i+1}. {col}")

  while True:
    try:
      other_choice = int(input("Enter your choice (1-{} or 'q' to quit): ".format(len(remaining_numeric_cols))))
      if 1 <= other_choice <= len(remaining_numeric_cols):
        return remaining_numeric_cols[other_choice - 1]
      elif other_choice.lower() == 'q':
        print("Scatter plot canceled.")
        return None
      else:
        print("Invalid choice. Please enter a number between 1 and {} or 'q' to quit.".format(len(remaining_numeric_cols)))
    except ValueError:
      print("Invalid input. Please enter a number.")


def visualize_categorical(data):
  """
  Prompts the user to choose a categorical feature and a visualization type.

  Args:
      data (pandas.DataFrame): The DataFrame containing the data.
  """
  categorical_cols = [col for col in data.columns if not pd.api.types.is_numeric_dtype(data[col])]

  if not categorical_cols:
    print("No categorical features found in the data.")
    return

  print("Choose a categorical feature to visualize:")
  for i, col in enumerate(categorical_cols):
    print(f"  {i+1}. {col}")

  while True:
    try:
      feature_choice = int(input("Enter your choice (1-{} or 'q' to quit): ".format(len(categorical_cols))))
      if 1 <= feature_choice <= len(categorical_cols):
        col = categorical_cols[feature_choice - 1]
        print("Choose a visualization type:")
        print("  1. Value Counts (shows the number of observations in each category)")  
        print("  2. Bar Chart (shows the number of observations in each category)")  
        print("  3. Pie Chart (shows the proportion of each category as pie slices, useful for few categories)")  
        print("  4. Histogram (shows the frequency distribution of categories)")  
        print("  'q' to quit")

        viz_choice = input("Enter your choice (1-4 or 'q'): ")
        if viz_choice.lower() == 'q':
          print("Exiting visualization.")
          return
        else:
          try:
            viz_choice = int(viz_choice)
            if 1 <= viz_choice <= 4:
              if viz_choice == 1:
                data[col].value_counts().plot(kind="bar")
                plt.xlabel(col)
                plt.ylabel("Number of observations")
                plt.title(f"Value Counts of {col}")
                plt.show()
              elif viz_choice == 2:
                data[col].value_counts().plot(kind="bar")
                plt.xlabel(col)
                plt.ylabel("Number of observations")
                plt.title(f"Bar Chart of {col}")
                plt.show()
              elif viz_choice == 3:
                data[col].value_counts().plot(kind="pie", autopct="%1.1f%%")
                plt.title(f"Pie Chart of {col}")
                plt.show()
              elif viz_choice == 4:
                data[col].value_counts().plot(kind="hist")
                plt.xlabel(col)
                plt.ylabel("Number of observations")
                plt.title(f"Histogram of {col}")
                plt.show()
              else:
                print("Invalid visualization choice.")
            else:
              print("Invalid visualization choice. Please choose between 1 and 4.")
          except ValueError:
            print("Invalid input. Please enter a number.")
      else:
        print("Invalid choice. Please enter a number between 1 and {} or 'q' to quit.".format(len(categorical_cols)))
    except:
      pass  # Placeholder to avoid needing a specific exception type


########################################################################################
# Correlation Analysis
########################################################################################
def analyze_correlations(data):
  """
  Analyzes correlations between numeric and categorical features in a DataFrame.

  Args:
      data (pandas.DataFrame): The DataFrame containing the data.
  """
  correlation_type = None
  while correlation_type not in ['pearson', 'spearman', 'no']:
    choice = input("Choose correlation type (pearson, spearman, or 'no' to skip): ").lower()
    if choice in ['pearson', 'spearman']:
      correlation_type = choice
    elif choice == 'no':
      print("Skipping correlation analysis.")
      return
    else:
      print("Invalid choice. Please choose 'pearson', 'spearman', or 'no'.")

  numeric_cols = data.select_dtypes(include=[np.number])
  categorical_cols = [col for col in data.columns if not pd.api.types.is_numeric_dtype(data[col])]

  # Analyze correlations between numeric features (if requested)
  if correlation_type in ['pearson', 'spearman']:
    if len(numeric_cols) > 1:
      print("Using", correlation_type, "correlation coefficient for numeric features.")
      print("  - Correlation coefficients closer to -1 and 1 indicate strong negative and positive relationships respectively.")
      print("  - Correlation coefficients closer to 0 indicate little to no relationship/association.")
      correlation_matrix = numeric_cols.corr(method=correlation_type)
      print(correlation_matrix)

  # Analyze associations between numeric and categorical features (using Chi-Square)
  for col in numeric_cols:
    for cat_col in categorical_cols:
      contingency_table = pd.crosstab(data[col], data[cat_col])
      chi2, pval, deg_of_freedom, expected_freq = chi2_contingency(contingency_table.values)
      print(f"Chi-Square Test between {col} and {cat_col}:")
      print(f"  p-value: {pval:.4f}")
      if pval < 0.05:
        print("  - Statistically significant association found.")
      else:
        print("  - No statistically significant association found.")

  # Analyze associations between categorical features (using Chi-Square)
  for cat_col1 in categorical_cols:
    for cat_col2 in categorical_cols:
      if cat_col1 != cat_col2:  # Avoid self-comparison
        contingency_table = pd.crosstab(data[cat_col1], data[cat_col2])
        chi2, pval, deg_of_freedom, expected_freq = chi2_contingency(contingency_table.values)
        print(f"Chi-Square Test between {cat_col1} and {cat_col2}:")
        print(f"  p-value: {pval:.4f}")
        if pval < 0.05:
          print("  - Statistically significant association found.")
        else:
          print("  - No statistically significant association found.")
