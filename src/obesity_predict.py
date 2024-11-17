# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import recall_score
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb
from sklearn.metrics import f1_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import os

# Step 1: Loading and Preprocessing Data
def load_and_preprocess_data(file_path):
    """
    Function to load and preprocess the dataset.
    """
    # Load the dataset
    data = pd.read_csv(file_path)
    
    # Rename columns for consistency
    data.columns = data.columns.str.lower().str.replace(' ', '_')

    print(data.dtypes)
    
    # Rename the target column
    data = data.rename(columns={'nobeyesdad': 'target_classification'})

    # Preserve original target labels for EDA
    data['original_target_classification'] = data['target_classification']
    
    # Encode categorical columns
    label_encoder = LabelEncoder()
    data['target_classification'] = label_encoder.fit_transform(data['target_classification'])
    
    # Encode other categorical features
    for column in data.select_dtypes(include=['object']).columns:
        data[column] = label_encoder.fit_transform(data[column])
    
    return data

# Step 2: Data Visualization and EDA
def visualize_data(data):
    """
    Function to visualize the data using basic plots.
    """
    # Plot the impact of Family History of Overweight on Obesity Levels
    plt.figure(figsize=(16, 10))
    sns.countplot(data=data, x='family_history_with_overweight', hue='original_target_classification', palette='coolwarm')
    plt.title('Impact of Family History of Overweight on Obesity Levels')
    plt.xlabel('Family History of Overweight')
    plt.ylabel('Count')
    plt.show()

    # Analyzing the impact of sleep duration (TUE) on obesity levels

    # Plot Sleep Duration (TUE) vs Obesity Levels
    plt.figure(figsize=(16, 10))
    sns.boxplot(data=data, x='tue', y='original_target_classification', palette='coolwarm')
    plt.title('Sleep Duration (TUE) vs Obesity Levels')
    plt.ylabel('Obesity Level')
    plt.xlabel('Sleep Duration (Hours per Day)')
    plt.xticks(rotation=0)
    plt.show()

    # Plot the effect of Eating High Caloric Food (FAVC) on Obesity Levels
    plt.figure(figsize=(16, 10))
    sns.countplot(data=data, x='favc', hue='original_target_classification', palette='magma')
    plt.title('Effect of Eating High Caloric Food (FAVC) on Obesity Levels')
    plt.xlabel('Eating High Caloric Food (Yes/No)')
    plt.ylabel('Count')
    plt.show()
    
    # Pairplot for visualizing relationships between variables
    sns.pairplot(data, hue='original_target_classification')
    plt.show()
    
    # Correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(data.corr(numeric_only = True), annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.show()

# Step 3: Feature Engineering (with PCA and Train/Validation/Test Split)
def feature_engineering_with_pca(data, n_components=2):
    """
    Function to scale, apply PCA, and split the dataset into training, validation, and test sets.
    """
    X = data.drop('target_classification', axis=1)  # Features
    y = data['target_classification']  # Target variable

    for column in X.select_dtypes(include=['object']).columns:
     le = LabelEncoder()
     X[column] = le.fit_transform(X[column])

    # Split the dataset into train (60%), validation (20%), and test (20%)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Scale the features using the training set statistics
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Apply PCA
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_val_pca = pca.transform(X_val_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    return X_train_pca, X_val_pca, X_test_pca, y_train, y_val, y_test

# Step 4: Model Building (XGBoost and Random Forest)

def train_logistic_regression(X_train_pca, y_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_pca, y_train)
    return model
    
def train_xgboost(X_train_pca, y_train):
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    model.fit(X_train_pca, y_train)
    return model

def train_random_forest(X_train_pca, y_train):
    model = RandomForestClassifier()
    model.fit(X_train_pca, y_train)
    return model

# Step 5: Hyperparameter Tuning
def hyperparameter_tuning_xgboost(X_train, y_train, X_val, y_val):
    """
    Function to perform hyperparameter tuning for XGBoost.
    """
    param_grid = {
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'n_estimators': [100, 200, 300]
    }

    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    
    val_score = grid_search.score(X_val, y_val)
    print(f"Best parameters for XGBoost: {grid_search.best_params_}, Validation Accuracy: {val_score}")
    
    return grid_search.best_estimator_

def hyperparameter_tuning_random_forest(X_train, y_train, X_val, y_val):
    """
    Function to perform hyperparameter tuning for Random Forest.
    """
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [5, 10, 15],
        'min_samples_split': [2, 5, 10]
    }

    model = RandomForestClassifier()
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    
    val_score = grid_search.score(X_val, y_val)
    print(f"Best parameters for Random Forest: {grid_search.best_params_}, Validation Accuracy: {val_score}")
    
    return grid_search.best_estimator_

# Step 6: Model Evaluation
def evaluate_model(model, X_test_pca, y_test):
    y_pred = model.predict(X_test_pca)
    print(classification_report(y_test, y_pred))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

# Main Function to Orchestrate Workflow
def main():
    # Path to the dataset
    file_path = os.path.join(os.path.dirname(__file__), '../data/Obesity_DataSet.csv')
    
    # Step 1: Load and preprocess the data
    data = load_and_preprocess_data(file_path)
    
    # Step 2: Visualize data and perform EDA
    visualize_data(data)
    
    # Step 3: Feature engineering with PCA (you can adjust n_components)
    X_train_pca, X_val_pca, X_test_pca, y_train, y_val, y_test = feature_engineering_with_pca(data, n_components=5)
    
    # Train and evaluate Logistic Regression
    logistic_model = train_logistic_regression(X_train_pca, y_train)
    print("Logistic Regression Model Performance on Test Set:")
    evaluate_model(logistic_model, X_test_pca, y_test)
    y_pred_logistic = logistic_model.predict(X_test_pca)  # Compute predictions for Logistic Regression
    
    # Train and evaluate XGBoost
    xgboost_model = train_xgboost(X_train_pca, y_train)
    print("XGBoost Model Performance on Test Set:")
    evaluate_model(xgboost_model, X_test_pca, y_test)
    y_pred_xgboost = xgboost_model.predict(X_test_pca)  # Compute predictions for XGBoost
    
    # Train and evaluate Random Forest
    random_forest_model = train_random_forest(X_train_pca, y_train)
    print("Random Forest Model Performance on Test Set:")
    evaluate_model(random_forest_model, X_test_pca, y_test)
    y_pred_rf = random_forest_model.predict(X_test_pca)  # Compute predictions for Random Forest
    
    # Plot recall scores for different classes
    plot_f1_score_for_classes(
        y_test,
        [y_pred_logistic, y_pred_xgboost, y_pred_rf],
        ["Logistic Regression", "XGBoost", "Random Forest"]
    )
    
def plot_f1_score_for_classes(y_true, model_predictions, model_names):
    """
    Plots F1 scores for each class across different models.
    
    Parameters:
        y_true (array-like): True labels.
        model_predictions (list of array-like): A list containing predicted labels from each model.
        model_names (list of str): Names of the models corresponding to the predictions.
    """
    # Calculate F1 scores for each model
    f1_scores = {}
    unique_classes = np.unique(y_true)

    for model_name, y_pred in zip(model_names, model_predictions):
        f1_per_class = f1_score(y_true, y_pred, average=None)
        f1_scores[model_name] = f1_per_class

    # Plot the F1 scores
    plt.figure(figsize=(12, 6))
    
    # Set the positions of the bars on the x-axis
    bar_width = 0.2
    bar_positions = np.arange(len(unique_classes))

    # Plot bars for each model
    for i, (model_name, f1s) in enumerate(f1_scores.items()):
        plt.bar(bar_positions + i * bar_width, f1s, width=bar_width, label=model_name)

    # Labeling
    plt.xlabel("Classes")
    plt.ylabel("F1 Score")
    plt.title("F1 Score for Different Classes Across Models")
    plt.xticks(bar_positions + bar_width * (len(model_names) / 2), unique_classes)
    plt.legend()
    plt.show()

# Run the main function
if __name__ == "__main__":
    main()
