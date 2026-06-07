import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_customer_data(n_customers=500):
    """Generate comprehensive customer data with timestamps"""
    
    np.random.seed(42)
    
    # Customer base features
    customers = []
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)  # 1 year of history
    
    for customer_id in range(1, n_customers + 1):
        # Random number of events per customer
        n_events = np.random.randint(5, 30)
        
        # Customer-level static features (same for all events of this customer)
        age = np.random.randint(18, 75)
        income = np.random.randint(30000, 200000)
        credit_score = np.random.randint(300, 850)
        account_age_days = np.random.randint(30, 3650)
        
        # Generate multiple snapshots for each customer (simulating historical observations)
        for i in range(n_events):
            timestamp = start_date + timedelta(
                days=np.random.randint(0, 365),
                hours=np.random.randint(0, 24)
            )
            
            # Features that can change over time
            num_products = np.random.randint(1, 6)
            has_credit_card = np.random.randint(0, 2)
            is_active_member = np.random.randint(0, 2)
            num_transactions_last_30d = np.random.randint(0, 80)
            avg_monthly_balance = np.random.uniform(100, 80000)
            
            # More realistic churn pattern: older accounts with low activity more likely to churn
            churn_probability = 0.1 + (0.3 if is_active_member == 0 else 0) + \
                               (0.2 if num_products <= 2 else 0) + \
                               (0.15 if num_transactions_last_30d < 10 else 0)
            churned = 1 if np.random.random() < churn_probability else 0
            
            customers.append({
                'customer_id': customer_id,
                'event_timestamp': timestamp,
                'age': age,
                'income': income,
                'credit_score': credit_score,
                'account_age_days': account_age_days,
                'num_products': num_products,
                'has_credit_card': has_credit_card,
                'is_active_member': is_active_member,
                'num_transactions_last_30d': num_transactions_last_30d,
                'avg_monthly_balance': avg_monthly_balance,
                'churned': churned
            })
    
    df = pd.DataFrame(customers)
    return df

def generate_transaction_data(customers_df, min_transactions=5, max_transactions=30):
    """Generate detailed transaction history features"""
    
    transactions = []
    start_date = datetime.now() - timedelta(days=180)  # 6 months of transactions
    
    # Get unique customers
    unique_customers = customers_df['customer_id'].unique()
    
    for customer_id in unique_customers:
        n_transactions = np.random.randint(min_transactions, max_transactions)
        
        # Get customer's active status for realistic transaction patterns
        customer_data = customers_df[customers_df['customer_id'] == customer_id].iloc[0]
        is_active = customer_data['is_active_member']
        
        for _ in range(n_transactions):
            timestamp = start_date + timedelta(
                days=np.random.randint(0, 180),
                hours=np.random.randint(0, 24)
            )
            
            # Active members have higher transaction amounts and frequency
            if is_active:
                transaction_amount = np.random.uniform(50, 5000)
                online_ratio_prob = 0.7
            else:
                transaction_amount = np.random.uniform(10, 1000)
                online_ratio_prob = 0.3
            
            transaction_type = np.random.choice(['online', 'in_store', 'atm'], 
                                               p=[online_ratio_prob, 0.2, 0.1])
            
            transactions.append({
                'customer_id': customer_id,
                'event_timestamp': timestamp,
                'transaction_date': timestamp.date(),
                'transaction_amount': round(transaction_amount, 2),
                'transaction_type': transaction_type,
                'merchant_category': np.random.choice(['retail', 'entertainment', 'groceries', 'travel', 'dining']),
                'is_weekend': 1 if timestamp.weekday() >= 5 else 0
            })
    
    return pd.DataFrame(transactions)

def generate_entity_dataframe_for_training(n_samples=1000):
    """Generate entity dataframe for training with timestamps"""
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)
    
    # Generate random timestamps for training examples
    timestamps = []
    customer_ids = []
    
    for i in range(n_samples):
        timestamp = start_date + timedelta(days=np.random.randint(0, 180))
        customer_id = np.random.randint(1, 501)  # Assuming 500 customers
        
        timestamps.append(timestamp)
        customer_ids.append(customer_id)
    
    entity_df = pd.DataFrame({
        "customer_id": customer_ids,
        "event_timestamp": timestamps
    })
    
    return entity_df

if __name__ == "__main__":
    # Generate and save data
    print("Generating customer data...")
    customers = generate_customer_data(500)
    customers.to_csv('data/raw_customer_features.csv', index=False)
    print(f"Generated {len(customers)} customer feature records")
    
    print("\nGenerating transaction data...")
    transactions = generate_transaction_data(customers.head(100), 10, 40)
    transactions.to_csv('data/raw_transaction_features.csv', index=False)
    print(f"Generated {len(transactions)} transaction records")
    
    print("\nGenerating entity dataframe for training...")
    entity_df = generate_entity_dataframe_for_training(1000)
    entity_df.to_csv('data/training_entity_df.csv', index=False)
    print(f"Generated {len(entity_df)} training entity records")import pandas as pd
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
