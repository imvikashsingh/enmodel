from feast import FeatureStore
import pandas as pd
from datetime import datetime, timedelta

class FeatureRetriever:
    """Class to retrieve features from the feature store"""
    
    def __init__(self):
        self.store = FeatureStore(repo_path=".")
    
    def get_features_for_prediction(self, customer_ids, timestamp=None):
        """Get latest features for prediction"""
        
        if timestamp is None:
            timestamp = datetime.now()
        
        # Define features to retrieve
        feature_refs = [
            "customer_demographic_features:age",
            "customer_demographic_features:income",
            "customer_demographic_features:credit_score",
            "customer_demographic_features:account_age_days",
            "customer_behavior_features:num_products",
            "customer_behavior_features:has_credit_card",
            "customer_behavior_features:is_active_member",
            "customer_behavior_features:num_transactions_last_30d",
            "customer_behavior_features:avg_monthly_balance",
            "transaction_summary_features:avg_transaction_amount",
            "transaction_summary_features:total_transaction_amount",
            "transaction_summary_features:online_transaction_ratio",
            "transaction_summary_features:weekend_transaction_ratio",
        ]
        
        # Retrieve features
        features = self.store.get_online_features(
            features=feature_refs,
            entity_rows=[{"customer": cust_id} for cust_id in customer_ids]
        ).to_dict()
        
        return pd.DataFrame(features)
    
    def get_historical_features(self, entity_df):
        """Get historical features for training"""
        
        feature_refs = [
            "customer_demographic_features:age",
            "customer_demographic_features:income",
            "customer_demographic_features:credit_score",
            "customer_demographic_features:account_age_days",
            "customer_behavior_features:num_products",
            "customer_behavior_features:has_credit_card",
            "customer_behavior_features:is_active_member",
            "customer_behavior_features:num_transactions_last_30d",
            "customer_behavior_features:avg_monthly_balance",
            "transaction_summary_features:avg_transaction_amount",
            "transaction_summary_features:total_transaction_amount",
            "transaction_summary_features:online_transaction_ratio",
            "transaction_summary_features:weekend_transaction_ratio",
            "churn_label:churned",
        ]
        
        historical_features = self.store.get_historical_features(
            entity_df=entity_df,
            features=feature_refs
        ).to_df()
        
        return historical_features
    
    def explore_feature_history(self, customer_id, feature_name, days=30):
        """View historical values of a specific feature"""
        
        # Create entity dataframe with timestamps
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        entity_df = pd.DataFrame({
            "customer": [customer_id] * len(dates),
            "event_timestamp": dates
        })
        
        # Retrieve historical features
        feature_refs = [f"customer_demographic_features:{feature_name}"]
        
        try:
            history = self.store.get_historical_features(
                entity_df=entity_df,
                features=feature_refs
            ).to_df()
            return history
        except Exception as e:
            print(f"Could not retrieve history for {feature_name}: {e}")
            return None

def display_features_for_customers(customer_ids):
    """Display features for specific customers"""
    
    retriever = FeatureRetriever()
    
    print("\n" + "="*60)
    print("FEATURES FOR CUSTOMER PREDICTION")
    print("="*60)
    
    # Get latest features
    features_df = retriever.get_features_for_prediction(customer_ids)
    
    print(f"\n📊 Features retrieved for {len(customer_ids)} customers:")
    print(features_df.to_string())
    
    # Show feature statistics
    print("\n📈 Feature Statistics:")
    for col in features_df.columns:
        if col != 'customer':
            print(f"  {col}: mean={features_df[col].mean():.2f}, "
                  f"min={features_df[col].min():.2f}, "
                  f"max={features_df[col].max():.2f}")
    
    return features_df

if __name__ == "__main__":
    # Test with some customer IDs
    test_customers = [1, 5, 10, 25, 50]
    display_features_for_customers(test_customers)
