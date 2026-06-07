import pandas as pd
import numpy as np
from feast import FeatureStore
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib
from mango.feature_transforms import prepare_features_for_training

class ChurnModelTrainer:
    """Train churn prediction model using features from Feast"""
    
    def __init__(self, feature_store_path="."):
        self.store = FeatureStore(repo_path=feature_store_path)
        self.model = None
        self.scaler = StandardScaler()
        
    def get_training_data(self, start_date=None, end_date=None):
        """Retrieve historical features for training"""
        
        if start_date is None:
            start_date = datetime.now() - timedelta(days=180)
        if end_date is None:
            end_date = datetime.now()
        
        print(f"📊 Retrieving training data from {start_date.date()} to {end_date.date()}")
        
        # Create entity dataframe with timestamps
        # In production, this would come from your event log
        customer_ids = list(range(1, 501))  # 500 customers
        dates = pd.date_range(start=start_date, end=end_date, freq='W')  # Weekly snapshots
        
        entity_rows = []
        for customer_id in customer_ids:
            for timestamp in dates:
                entity_rows.append({
                    "customer_id": customer_id,
                    "event_timestamp": timestamp
                })
        
        entity_df = pd.DataFrame(entity_rows)
        
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
            "transaction_summary_features:max_transaction_amount",
            "transaction_summary_features:total_transaction_amount",
            "transaction_summary_features:transaction_count",
            "transaction_summary_features:online_transaction_ratio",
            "transaction_summary_features:weekend_transaction_ratio",
            "churn_label:churned",
            "customer_risk_features:credit_score_category",
            "customer_risk_features:account_age_category",
            "customer_risk_features:is_high_risk"
        ]
        
        print("🔄 Retrieving historical features from Feast...")
        training_data = self.store.get_historical_features(
            entity_df=entity_df,
            features=feature_refs
        ).to_df()
        
        print(f"✅ Retrieved {len(training_data)} training examples")
        return training_data
    
    def prepare_data(self, training_data):
        """Prepare features and target for model training"""
        
        # Separate features and target
        target_col = 'churned'
        
        if target_col not in training_data.columns:
            raise ValueError(f"Target column '{target_col}' not found in training data")
        
        X = training_data.drop(columns=[target_col, 'customer_id', 'event_timestamp'])
        y = training_data[target_col]
        
        # Remove any other non-feature columns
        exclude_cols = ['event_timestamp']
        X = X.drop(columns=[col for col in exclude_cols if col in X.columns])
        
        # Add derived features
        print("🔄 Adding derived features...")
        X = prepare_features_for_training(X)
        
        # Handle missing values
        X = X.fillna(0)
        
        return X, y
    
    def train(self, X_train, y_train):
        """Train the churn prediction model"""
        
        print("🤖 Training Random Forest model...")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train model
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            class_weight='balanced'
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        print("✅ Model training completed!")
        return self.model
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        
        X_test_scaled = self.scaler.transform(X_test)
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        print("\n" + "="*60)
        print("📊 MODEL EVALUATION RESULTS")
        print("="*60)
        
        print(f"\nROC-AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Not Churned', 'Churned']))
        
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X_test.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\n🔝 Top 10 Most Important Features:")
        print(feature_importance.head(10).to_string(index=False))
        
        return {
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'feature_importance': feature_importance
        }
    
    def save_model(self, path='models/churn_model.pkl'):
        """Save trained model and scaler"""
        
        import os
        os.makedirs('models', exist_ok=True)
        
        model_artifacts = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.model.feature_names_in_ if hasattr(self.model, 'feature_names_in_') else None
        }
        
        joblib.dump(model_artifacts, path)
        print(f"✅ Model saved to {path}")
        
    def load_model(self, path='models/churn_model.pkl'):
        """Load trained model"""
        
        model_artifacts = joblib.load(path)
        self.model = model_artifacts['model']
        self.scaler = model_artifacts['scaler']
        print(f"✅ Model loaded from {path}")

def run_training_pipeline():
    """Complete training pipeline using feature store"""
    
    print("\n" + "="*60)
    print("🎯 CHURN PREDICTION MODEL TRAINING PIPELINE")
    print("="*60)
    
    # Initialize trainer
    trainer = ChurnModelTrainer()
    
    # Get training data from Feast
    training_data = trainer.get_training_data()
    
    # Prepare features
    X, y = trainer.prepare_data(training_data)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n📊 Data Split:")
    print(f"  Training set: {len(X_train)} samples")
    print(f"  Test set: {len(X_test)} samples")
    print(f"  Features: {len(X_train.columns)}")
    
    # Train model
    trainer.train(X_train, y_train)
    
    # Evaluate
    metrics = trainer.evaluate(X_test, y_test)
    
    # Save model
    trainer.save_model()
    
    return trainer

if __name__ == "__main__":
    trainer = run_training_pipeline()
