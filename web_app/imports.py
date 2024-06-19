from cleaning import read_data, handle_missing_values, identify_and_handle_outliers, handle_duplicates, handle_formatting
from eda import visualize_numerical, visualize_categorical, analyze_correlations
from preprocessing import convert_data_types, scale, create_interaction_feature, create_feature_bins, create_custom_features, create_one_hot_encoding, create_label_encoding, handle_class_imbalance, feature_selection
from modeling import model_selection, data_splitting, train_model, tune_hyperparameters, evaluate_model
from analysis import generate_classification_report, visualize_confusion_matrix, plot_learning_curves, plot_roc_curve, plot_precision_recall_curve, explain_with_shap, plot_partial_dependence, analyze_feature_importance, save_model

# Imports from cleaning
import pandas as pd
import json
import numpy as np

# Imports from eda
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
import numpy as np

# Imports from preprocessing
import pandas as pd
import numpy as np 
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import sympy as sp
import sklearn
from imblearn.under_sampling import RandomUnderSampler  # Import for undersampling
from imblearn.over_sampling import SMOTE  # Import for oversampling
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2  # Example statistical test

# Imports from modeling
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Imports from analysis
import seaborn as sns
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
