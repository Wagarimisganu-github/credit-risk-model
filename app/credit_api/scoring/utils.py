# scoring/utils.py
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

def encode_categorical_features(df, categorical_columns):
    """Encode categorical features using one-hot encoding."""
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded_features = encoder.fit_transform(df[categorical_columns])
    feature_names = encoder.get_feature_names_out(categorical_columns)
    encoded_df = pd.DataFrame(encoded_features, columns=feature_names, index=df.index)
    return pd.concat([df.drop(columns=categorical_columns), encoded_df], axis=1)

def handle_missing_values(df):
    num_imputer = SimpleImputer(strategy='median')
    numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
    df[numerical_columns] = num_imputer.fit_transform(df[numerical_columns])
    
    cat_imputer = SimpleImputer(strategy='most_frequent')
    categorical_columns = df.select_dtypes(include=['object']).columns
    df[categorical_columns] = cat_imputer.fit_transform(df[categorical_columns])
    
    return df

def normalize_features(df):
    scaler = StandardScaler()
    numerical_columns = df.select_dtypes(include=['int64', 'int32', 'float64']).columns
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
    return df

# def rfms_score(df):
#     """Calculate RFMS score for each customer."""
#     current_date = df['TransactionStartTime'].max()
    
#     customer_metrics = df.groupby('CustomerId').agg({
#         'TransactionStartTime': lambda x: (current_date - x.max()).days,  # Recency
#         'TransactionId': 'count',  # Frequency
#         'Amount': ['sum', 'mean'],  # Monetary and Size
#     })
    
#     customer_metrics.columns = ['Recency', 'Frequency', 'MonetaryTotal', 'MonetaryAvg']
    
#     # Normalize the metrics
#     for col in customer_metrics.columns:
#         customer_metrics[f'{col}_Normalized'] = (customer_metrics[col] - customer_metrics[col].min()) / (customer_metrics[col].max() - customer_metrics[col].min())
    
#     # Calculate RFMS score
#     customer_metrics['RFMS_Score'] = (
#         0.25 * (1 - customer_metrics['Recency_Normalized']) +  # Inverse of Recency
#         0.25 * customer_metrics['Frequency_Normalized'] +
#         0.25 * customer_metrics['MonetaryTotal_Normalized'] +
#         0.25 * customer_metrics['MonetaryAvg_Normalized']
#     )
    
#     return customer_metrics
