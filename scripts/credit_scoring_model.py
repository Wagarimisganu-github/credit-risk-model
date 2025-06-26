# credit_scoring_utils.py
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
import matplotlib.pyplot as plt
import joblib
import pickle

def split_data(X, y, test_size=0.2, random_state=42, stratify=True):
    if stratify:
        return train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )
    else:
        return train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state
        )

def create_models():
    """
    Create logistic regression, decision tree, random forest, and gradient boosting models.
    """
    logistic_model = LogisticRegression(random_state=42)
    dt_model = DecisionTreeClassifier(random_state=42)
    rf_model = RandomForestClassifier(random_state=42)
    gb_model = GradientBoostingClassifier(random_state=42)
    return logistic_model, dt_model, rf_model, gb_model

def train_model(model, X_train, y_train):
    """
    Train a given model on the training data.
    """
    model.fit(X_train, y_train)
    return model

def tune_model(model, X_train, y_train, param_grid):
    """
    Perform hyperparameter tuning for a given model using GridSearchCV.
    """
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_
    
def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model and return various performance metrics.
    """
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label='good')
    recall = recall_score(y_test, y_pred, pos_label='good')
    f1 = f1_score(y_test, y_pred, pos_label='good')
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc
    }

def plot_roc_curve(model, X_test, y_test):
    """
    Plot the ROC curve for a given model.
    """
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba, pos_label='good')
    
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc_score(y_test, y_pred_proba):.4f})')
    plt.plot([0, 1], [0, 1], linestyle='--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()

def save_model(model, filename):
    """
    Save the trained model to a file.
    """
    with open(filename, 'wb') as file:
        pickle.dump(model, file)
    print(f"Model saved to {filename}")

def load_model(filename):
    """
    Load a trained model from a file.
    """
    return joblib.load(filename)

def plot_confusion_matrix(model, X_test, y_test):
    """
    Plot the confusion matrix for a given model.
    """
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

def plot_learning_curve(model, X, y, cv=5):
    """
    Plot the learning curve for a given model.
    """
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=cv, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10))
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, train_mean, label='Training score')
    plt.plot(train_sizes, test_mean, label='Cross-validation score')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1)
    plt.xlabel('Number of training examples')
    plt.ylabel('Score')
    plt.title('Learning Curve')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()


def analyze_feature_correlations(X):
    """
    Analyze and plot feature correlations.
    """
    corr_matrix = X.corr()
    plt.figure(figsize=(10, 10))
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm')
    plt.title('Feature Correlation Heatmap')
    plt.show()
    
    # Print highly correlated feature pairs
    high_corr = (corr_matrix.abs() > 0.8) & (corr_matrix != 1.0)
    high_corr_pairs = [(corr_matrix.index[i], corr_matrix.columns[j])
                       for i in range(len(corr_matrix.index))
                       for j in range(i+1, len(corr_matrix.columns))
                       if high_corr.iloc[i, j]]
    if high_corr_pairs:
        print("Highly correlated feature pairs:")
        for pair in high_corr_pairs:
            print(f"{pair[0]} - {pair[1]}: {corr_matrix.loc[pair[0], pair[1]]:.2f}")
    else:
        print("No highly correlated feature pairs found.")


def plot_feature_importance(model, X):
    """
    Plot feature importance for tree-based models.
    """
    if hasattr(model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        plt.figure(figsize=(10, 6))
        feature_importance.head(10).plot(x='feature', y='importance', kind='bar')
        plt.title('Top 10 Feature Importances')
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
    else:
        print("This model doesn't have feature importances.")