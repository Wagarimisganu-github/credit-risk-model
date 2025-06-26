import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(file_path):
    """Load the dataset from a CSV file."""
    return pd.read_csv(file_path, index_col=False)

def data_overview(df):
    """Provide an overview of the dataset."""
    print("Dataset Shape:", df.shape)
    print("\nColumn Names:", df.columns.tolist())
    print("\nData Types:")
    print(df.dtypes)
    print("\nMissing Values:")
    print(df.isnull().sum())

def summary_statistics(df):
    """Calculate and return summary statistics for numerical columns."""
    return df.describe()

def plot_numerical_distributions(df):
    """Plot distributions of numerical features."""
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    n_cols = 2
    n_rows = (len(numerical_cols) + 1) // 2  # ensure enough rows for all plots
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axes = axes.flatten()

    for i, col in enumerate(numerical_cols):
        sns.histplot(df[col], kde=True, ax=axes[i], color='blue')
        axes[i].set_title(f'Distribution of {col}')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Count')
    
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


def plot_categorical_distributions(df, columns):
    """Plot distributions of specified categorical features."""
    categorical_cols = [col for col in columns if df[col].dtype == 'object']  # Filter only categorical columns
    n_cols = 2
    n_rows = (len(categorical_cols) + 1) // 2  # ensure enough rows for all plots
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axes = axes.flatten()

    for i, col in enumerate(categorical_cols):
        df[col].value_counts().plot(kind='bar', ax=axes[i])
        axes[i].set_title(f'Distribution of {col}')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Count')
        axes[i].tick_params(axis='x', rotation=45)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


def correlation_analysis(df):
    """Perform correlation analysis on numerical features."""
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    corr_matrix = df[numerical_cols].corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Matrix of Numerical Features')
    plt.show()

def detect_outliers(df):
    """Detect outliers using box plots for numerical features."""
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    n_cols = 2
    n_rows = (len(numerical_cols) + 1) // 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axes = axes.flatten()

    for i, col in enumerate(numerical_cols):
        sns.boxplot(x=df[col], ax=axes[i])
        axes[i].set_title(f'Box Plot of {col}')
        axes[i].set_xlabel(col)
    
    # Remove unused axes 
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()



def bivariate_analysis(df):
    """
    Perform bivariate analysis and generate related plots.
    """
    # 1. Amount vs Product Category
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='ProductCategory', y='Amount', data=df)
    plt.title('Transaction Amount by Product Category')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    plt.close()

    # 2. Fraud Analysis
    # Fraud vs ProductCategory
    fraud_rate = df.groupby('ProductCategory')['FraudResult'].mean()
    plt.figure(figsize=(12, 6))
    fraud_rate.sort_values(ascending=False).plot(kind='bar')
    plt.title('Fraud Rate by Product Category')
    plt.ylabel('Fraud Rate')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    plt.close()

    # 3. Fraud vs Amount
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='FraudResult', y='Amount', data=df)
    plt.title('Transaction Amount for Fraudulent vs Non-Fraudulent Transactions')
    plt.tight_layout()
    plt.show()
    plt.close()

    # 4. ChannelId vs Amount
    channel_avg_amount = df.groupby('ChannelId')['Amount'].mean().sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    channel_avg_amount.plot(kind='bar')
    plt.title('Average Transaction Amount by Channel')
    plt.ylabel('Average Amount')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    plt.close()

    # 5. Customer Analysis
    customer_transactions = df.groupby('CustomerId')['Amount'].agg(['count', 'sum', 'mean'])
    customer_transactions = customer_transactions.sort_values('sum', ascending=False)

    plt.figure(figsize=(12, 6))
    plt.scatter(customer_transactions['count'], customer_transactions['sum'])
    plt.title('Customer Transaction Count vs Total Amount')
    plt.xlabel('Number of Transactions')
    plt.ylabel('Total Transaction Amount')
    plt.show()
    plt.close()

    # 6. Transaction Amount Analysis
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='TransactionStartTime', y='Amount', data=df)
    plt.title('Transaction Amount over Time')
    plt.xlabel('Transaction Start Time')
    plt.ylabel('Amount')
    plt.xticks(rotation=45)
    plt.show()
    plt.close()


def multivariate_analysis(df):
    """
    Perform multivariate analysis and generate related plots.
    """
    # 1. Fraud Analysis across Multiple Features
    # Fraud by Product Category and Channel
    fraud_by_category_channel = df.groupby(['ProductCategory', 'ChannelId'])['FraudResult'].mean().unstack()
    plt.figure(figsize=(14, 8))
    sns.heatmap(fraud_by_category_channel, annot=True, cmap='YlOrRd')
    plt.title('Fraud Rate by Product Category and Channel')
    plt.tight_layout()
    plt.show()
    plt.close()

    # 2. Fraud by Date and Amount
    df['Date'] = df['TransactionStartTime'].dt.date
    fraud_by_date = df.groupby('Date').agg({'FraudResult': 'mean', 'Amount': 'mean'})
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()
    ax1.plot(fraud_by_date.index, fraud_by_date['FraudResult'], 'g-', label='Fraud Rate')
    ax2.plot(fraud_by_date.index, fraud_by_date['Amount'], 'b-', label='Average Amount')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Fraud Rate', color='g')
    ax2.set_ylabel('Average Amount', color='b')
    plt.title('Fraud Rate and Average Amount Over Time')
    fig.legend(loc="upper left", bbox_to_anchor=(0.1,0.9), bbox_transform=ax1.transAxes)
    plt.tight_layout()
    plt.show()
    plt.close()


    # 3. Product Category, Amount, and Fraud Interaction
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    categories = df['ProductCategory'].unique()
    colors = plt.cm.rainbow(np.linspace(0, 1, len(categories)))
    for category, color in zip(categories, colors):
        category_data = df[df['ProductCategory'] == category]
        ax.scatter(category_data['Amount'], category_data['FraudResult'], 
                   category_data.index, c=[color], label=category, alpha=0.6)
    ax.set_xlabel('Amount')
    ax.set_ylabel('Fraud Result')
    ax.set_zlabel('Transaction Index')
    plt.title('3D Scatter Plot: Amount, Fraud, and Product Category')
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.close()


def time_series_analysis(df):
    """
    Perform time series analysis and generate related plots.
    """
    # daily transaction volume
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
    df['Date'] = df['TransactionStartTime'].dt.date
    daily_transactions = df.groupby('Date')['Amount'].sum().reset_index()

    plt.figure(figsize=(12, 6))
    plt.plot(daily_transactions['Date'], daily_transactions['Amount'])
    plt.title('Daily Transaction Volume')
    plt.xlabel('Date')
    plt.ylabel('Total Amount')
    plt.xticks(rotation=45)
    plt.show()
    plt.close()

    
    # Time of Day Analysis
    df['Hour'] = df['TransactionStartTime'].dt.hour
    hourly_transactions = df.groupby('Hour').size()
    plt.figure(figsize=(12, 6))
    hourly_transactions.plot(kind='bar')
    plt.title('Number of Transactions by Hour of Day')
    plt.xlabel('Hour')
    plt.ylabel('Number of Transactions')
    plt.tight_layout()
    plt.show()
    plt.close()

    # Day of the Week Analysis
    df['DayOfWeek'] = df['TransactionStartTime'].dt.dayofweek
    daily_transactions = df.groupby('DayOfWeek').size()
    plt.figure(figsize=(10, 6))
    daily_transactions.plot(kind='bar')
    plt.title('Number of Transactions by Day of Week')
    plt.xlabel('Day of Week (0=Monday, 6=Sunday)')
    plt.ylabel('Number of Transactions')
    plt.tight_layout()
    plt.show()
    plt.close()

    # Trend Analysis
    monthly_transactions = df.groupby(df['TransactionStartTime'].dt.to_period('M')).agg({
        'TransactionId': 'count',
        'Amount': 'sum'
    })
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()
    ax1.plot(monthly_transactions.index.astype(str), monthly_transactions['TransactionId'], 'g-', label='Number of Transactions')
    ax2.plot(monthly_transactions.index.astype(str), monthly_transactions['Amount'], 'b-', label='Total Amount')
    ax1.set_xlabel('Month')
    ax1.set_ylabel('Number of Transactions', color='g')
    ax2.set_ylabel('Total Amount', color='b')
    plt.title('Monthly Transaction Volume and Total Amount')
    plt.xticks(rotation=45)
    fig.legend(loc="upper left", bbox_to_anchor=(0.1,0.9), bbox_transform=ax1.transAxes)
    plt.tight_layout()
    plt.show()
    plt.close()