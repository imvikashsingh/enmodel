import os
import sys
import subprocess
from datetime import datetime

def setup_environment():
    """Setup necessary directories and install dependencies"""
    
    # Create directories
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    # Install required packages
    print("📦 Installing required packages...")
    packages = [
        "feast",
        "pandas",
        "numpy",
        "duckdb",
        "scikit-learn",
        "joblib",
        "pydantic",
        "google-cloud-bigquery",
        "google-cloud-storage"
    ]
    
    for package in packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])
            print(f"  ✓ {package}")
        except:
            print(f"  ⚠️ {package} may already be installed")
    
    print("✅ Environment setup complete")

def run_complete_demo():
    """Run complete feature store and training demo"""
    
    print("\n" + "="*70)
    print("🏦 MANGO FEATURE STORE - BANK CHURN PREDICTION SYSTEM")
    print("="*70)
    
    # Step 1: Generate data
    print("\n📊 STEP 1: Generating synthetic bank customer data...")
    from mango.data_generator import generate_customer_data, generate_transaction_data
    
    customers_df = generate_customer_data(500)
    customers_df.to_csv('data/raw_customer_features.csv', index=False)
    print(f"  ✓ Generated {len(customers_df)} customer records")
    
    transactions_df = generate_transaction_data(customers_df, 10, 40)
    transactions_df.to_csv('data/raw_transaction_features.csv', index=False)
    print(f"  ✓ Generated {len(transactions_df)} transaction records")
    
    # Step 2: Apply Feast features
    print("\n🏗️ STEP 2: Building feature store...")
    from feast import FeatureStore
    store = FeatureStore(repo_path=".")
    
    # Step 3: Materialize features
    print("\n💾 STEP 3: Materializing features to DuckDB...")
    from mango.materialize_features import materialize_all_features
    materialize_all_features()
    
    # Step 4: Train model
    print("\n🤖 STEP 4: Training churn prediction model...")
    from mango.train_model import run_training_pipeline
    trainer = run_training_pipeline()
    
    # Step 5: Make predictions
    print("\n🔮 STEP 5: Making predictions on new customers...")
    from mango.predict_churn import run_prediction_demo
    predictions = run_prediction_demo()
    
    print("\n" + "="*70)
    print("✨ DEMO COMPLETED SUCCESSFULLY!")
    print("="*70)
    
    print("\n📚 What you've learned:")
    print("  1. Feature Store Setup - Using Feast with DuckDB backend")
    print("  2. Feature Definition - Creating entities, feature views, and sources")
    print("  3. Feature Materialization - Moving features to online store")
    print("  4. Training Pipeline - Retrieving historical features from Feast")
    print("  5. Inference Pipeline - Getting online features for predictions")
    print("  6. Model Tracking - Feature lineage and versioning")
    
    print("\n💡 Next Steps:")
    print("  • Add more complex feature transformations")
    print("  • Implement feature validation and monitoring")
    print("  • Set up automated feature materialization jobs")
    print("  • Add real-time feature streaming")
    
    return predictions

if __name__ == "__main__":
    setup_environment()
    run_complete_demo()
