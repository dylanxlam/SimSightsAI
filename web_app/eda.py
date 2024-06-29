########################################################################################
# Import Statements
########################################################################################
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
import numpy as np


########################################################################################
# Descriptive Statistics
########################################################################################

########################################################################################
# Data Distribution Visualization
########################################################################################
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def visualize_numerical(data):
  """
  Guides the user through visualizing numerical features.

  Args:
      data (pandas.DataFrame): The DataFrame containing the data.
  """
  numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
  
  if not numeric_cols:
    print("No numeric features found in the data.")
    return

  while True:
    print("\nNumeric features you can visualize:")
    for i, col in enumerate(numeric_cols, 1):
      print(f"  {i}. {col}")
    print("  q. Quit")

    choice = input("Enter your choice (number or 'q' to quit): ").lower()
    
    if choice == 'q':
      break
    
    try:
      col_index = int(choice) - 1
      if 0 <= col_index < len(numeric_cols):
        col = numeric_cols[col_index]
        visualize_feature(data, col, is_numeric=True)
      else:
        print("Invalid choice. Please try again.")
    except ValueError:
      print("Invalid input. Please enter a number or 'q'.")

def visualize_categorical(data):
  """
  Guides the user through visualizing categorical features.

  Args:
      data (pandas.DataFrame): The DataFrame containing the data.
  """
  categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
  
  if not categorical_cols:
    print("No categorical features found in the data.")
    return

  while True:
    print("\nCategorical features you can visualize:")
    for i, col in enumerate(categorical_cols, 1):
      print(f"  {i}. {col}")
    print("  q. Quit")

    choice = input("Enter your choice (number or 'q' to quit): ").lower()
    
    if choice == 'q':
      break
    
    try:
      col_index = int(choice) - 1
      if 0 <= col_index < len(categorical_cols):
        col = categorical_cols[col_index]
        visualize_feature(data, col, is_numeric=False)
      else:
        print("Invalid choice. Please try again.")
    except ValueError:
      print("Invalid input. Please enter a number or 'q'.")

def visualize_feature(data, col, is_numeric):
  """
  Visualizes a single feature based on user's choice of plot type.

  Args:
      data (pandas.DataFrame): The DataFrame containing the data.
      col (str): The name of the column to visualize.
      is_numeric (bool): Whether the feature is numeric or categorical.
  """
  while True:
    print(f"\nChoose a visualization type for {col}:")
    if is_numeric:
      print("  1. Histogram")
      print("  2. Box Plot")
      print("  3. Violin Plot")
      print("  4. Kernel Density Plot")
    else:
      print("  1. Bar Plot")
      print("  2. Count Plot")
      print("  3. Pie Chart")
    print("  q. Back to feature selection")

    choice = input("Enter your choice: ").lower()

    if choice == 'q':
      break

    try:
      choice = int(choice)
      plt.figure(figsize=(10, 6))
      
      if is_numeric:
        print("Close the plot window to continue with this program.")
        if choice == 1:
          sns.histplot(data=data, x=col, kde=True)
          plt.title(f"Histogram of {col}")
        elif choice == 2:
          sns.boxplot(data=data, y=col)
          plt.title(f"Box Plot of {col}")
        elif choice == 3:
          sns.violinplot(data=data, y=col)
          plt.title(f"Violin Plot of {col}")
        elif choice == 4:
          sns.kdeplot(data=data[col])
          plt.title(f"Kernel Density Plot of {col}")
        else:
          print("Invalid choice. Please try again.")
          continue
      else:
        if choice == 1:
          print("Close the plot window to continue with this program.")
          sns.barplot(data=data, x=col, y=data[col].value_counts().index, orient='h')
          plt.title(f"Bar Plot of {col}")
        elif choice == 2:
          sns.countplot(data=data, y=col)
          plt.title(f"Count Plot of {col}")
        elif choice == 3:
          data[col].value_counts().plot(kind='pie', autopct='%1.1f%%')
          plt.title(f"Pie Chart of {col}")
        else:
          print("Invalid choice. Please try again.")
          continue

      plt.tight_layout()
      plt.show()
    except ValueError:
      print("Invalid input. Please enter a number or 'q'.")

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
