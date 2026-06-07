from feast import FeatureStore
import pandas as pd
from datetime import datetime, timedelta

def materialize_all_features():
    """Materialize features from offline to online store using DuckDB"""
    
    # Initialize feature store
    store = FeatureStore(repo_path=".")
    
    # Materialize features for last 365 days
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    print(f"🔄 Materializing features from {start_date.date()} to {end_date.date()}")
    
    try:
        store.materialize(
            feature_views=[
                "customer_demographic_features",
                "customer_behavior_features",
                "transaction_summary_features",
                "churn_label",
                "customer_risk_features"
            ],
            start_date=start_date,
            end_date=end_date
        )
        print("✅ Features materialized successfully to DuckDB!")
        
        # Verify materialization
        print("\n📊 Verifying online store...")
        sample_customers = [1, 10, 50, 100]
        
        for customer_id in sample_customers:
            features = store.get_online_features(
                features=[
                    "customer_demographic_features:credit_score",
                    "customer_behavior_features:is_active_member",
                    "churn_label:churned"
                ],
                entity_rows=[{"customer_id": customer_id}]
            ).to_dict()
            print(f"  Customer {customer_id}: {features}")
            
    except Exception as e:
        print(f"❌ Error during materialization: {e}")

def get_feature_store_metadata():
    """Display feature store metadata"""
    
    store = FeatureStore(repo_path=".")
    
    print("\n" + "="*60)
    print("🏪 FEATURE STORE METADATA")
    print("="*60)
    
    # List all entities
    entities = store.list_entities()
    print(f"\n📊 Entities ({len(entities)}):")
    for entity in entities:
        print(f"  - {entity.name}: {entity.description}")
    
    # List all feature views
    feature_views = store.list_feature_views()
    print(f"\n📈 Feature Views ({len(feature_views)}):")
    for fv in feature_views:
        print(f"  - {fv.name}")
        print(f"    Features: {[f.name for f in fv.features]}")
        print(f"    TTL: {fv.ttl}")
        print(f"    Tags: {fv.tags}")
    
    return store

if __name__ == "__main__":
    get_feature_store_metadata()
    materialize_all_features()
