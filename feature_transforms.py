import pandas as pd
import numpy as np

def add_derived_features(df):
    """Add derived features for better model performance"""
    
    df = df.copy()
    
    # Credit score categories
    df['credit_score_category'] = pd.cut(df['credit_score'], 
                                         bins=[0, 580, 670, 740, 800, 850],
                                         labels=['Very Poor', 'Poor', 'Fair', 'Good', 'Excellent'])
    
    # Account age categories
    df['account_age_category'] = pd.cut(df['account_age_days'],
                                        bins=[0, 365, 730, 1825, 3650],
                                        labels=['New', 'Established', 'Loyal', 'Long-term'])
    
    # Risk score (composite)
    df['is_high_risk'] = (
        (df['credit_score'] < 600) |
        (df['is_active_member'] == 0) |
        (df['num_products'] <= 1) |
        (df['num_transactions_last_30d'] < 5)
    ).astype(int)
    
    # Transaction behavior score
    if 'avg_transaction_amount' in df.columns:
        df['high_value_customer'] = (df['avg_transaction_amount'] > df['avg_transaction_amount'].median()).astype(int)
    
    # Balance to income ratio
    if 'avg_monthly_balance' in df.columns and 'income' in df.columns:
        df['balance_to_income_ratio'] = df['avg_monthly_balance'] / (df['income'] / 12)
        df['balance_to_income_ratio'] = df['balance_to_income_ratio'].fillna(0).clip(0, 5)
    
    return df

def prepare_features_for_training(features_df):
    """Prepare features for model training"""
    
    # Add derived features
    features_df = add_derived_features(features_df)
    
    # Encode categorical variables
    categorical_cols = ['credit_score_category', 'account_age_category']
    for col in categorical_cols:
        if col in features_df.columns:
            features_df[col] = features_df[col].astype('category').cat.codes
    
    # Select final feature columns
    feature_columns = [
        'age', 'income', 'credit_score', 'account_age_days',
        'num_products', 'has_credit_card', 'is_active_member',
        'num_transactions_last_30d', 'avg_monthly_balance',
        'avg_transaction_amount', 'max_transaction_amount', 
        'total_transaction_amount', 'transaction_count',
        'online_transaction_ratio', 'weekend_transaction_ratio',
        'credit_score_category', 'account_age_category', 'is_high_risk'
    ]
    
    # Only keep columns that exist in the dataframe
    available_features = [col for col in feature_columns if col in features_df.columns]
    
    return features_df[available_features]
