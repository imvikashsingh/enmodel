import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_customer_data(n_customers=100):
    """Generate sample customer data with timestamps"""
    
    np.random.seed(42)
    
    # Customer base features
    customers = []
    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)
    
    for customer_id in range(1, n_customers + 1):
        # Random number of transactions per customer
        n_transactions = np.random.randint(1, 20)
        
        for i in range(n_transactions):
            timestamp = start_date + timedelta(
                days=np.random.randint(0, 180),
                hours=np.random.randint(0, 24)
            )
            
            customers.append({
                'customer_id': customer_id,
                'event_timestamp': timestamp,
                'age': np.random.randint(18, 70),
                'income': np.random.randint(30000, 150000),
                'credit_score': np.random.randint(300, 850),
                'account_age_days': np.random.randint(30, 3650),
                'num_products': np.random.randint(1, 5),
                'has_credit_card': np.random.randint(0, 2),
                'is_active_member': np.random.randint(0, 2),
                'transaction_amount': np.random.uniform(10, 5000),
                'num_transactions_last_30d': np.random.randint(0, 50),
                'avg_monthly_balance': np.random.uniform(100, 50000),
                'churned': np.random.choice([0, 1], p=[0.8, 0.2])  # Target label
            })
    
    df = pd.DataFrame(customers)
    return df

def generate_transaction_data(customers_df, n_transactions_per_customer=10):
    """Generate transaction history features"""
    
    transactions = []
    start_date = datetime.now() - timedelta(days=90)
    
    for _, customer in customers_df.iterrows():
        for _ in range(n_transactions_per_customer):
            timestamp = start_date + timedelta(
                days=np.random.randint(0, 90),
                hours=np.random.randint(0, 24)
            )
            
            transactions.append({
                'customer_id': customer['customer_id'],
                'event_timestamp': timestamp,
                'transaction_date': timestamp,
                'transaction_amount': np.random.uniform(10, 5000),
                'transaction_type': np.random.choice(['online', 'in_store', 'atm']),
                'merchant_category': np.random.choice(['retail', 'entertainment', 'groceries', 'travel']),
                'is_weekend': 1 if timestamp.weekday() >= 5 else 0
            })
    
    return pd.DataFrame(transactions)

if __name__ == "__main__":
    # Generate and save data
    customers = generate_customer_data(200)
    customers.to_csv('data/raw_customer_features.csv', index=False)
    print(f"Generated {len(customers)} customer feature records")
    
    transactions = generate_transaction_data(customers.head(50), 20)
    transactions.to_csv('data/raw_transaction_features.csv', index=False)
    print(f"Generated {len(transactions)} transaction records")
