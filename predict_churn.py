import pandas as pd
import numpy as np
from feast import FeatureStore
import joblib
from datetime import datetime
from mango.feature_transforms import prepare_features_for_training

class ChurnPredictor:
    """Make churn predictions using features from Feast"""
    
    def __init__(self, model_path='models/churn_model.pkl', feature_store_path="."):
        # Load model
        model_artifacts = joblib.load(model_path)
        self.model = model_artifacts['model']
        self.scaler = model_artifacts['scaler']
        
        # Initialize feature store
        self.store = FeatureStore(repo_path=feature_store_path)
        
    def get_features_for_customers(self, customer_ids):
        """Get latest features for customers from online store"""
        
        print(f"🔍 Retrieving features for {len(customer_ids)} customers...")
        
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
            "customer_risk_features:credit_score_category",
            "customer_risk_features:account_age_category",
            "customer_risk_features:is_high_risk"
        ]
        
        entity_rows = [{"customer_id": cust_id} for cust_id in customer_ids]
        
        features_dict = self.store.get_online_features(
            features=feature_refs,
            entity_rows=entity_rows
        ).to_dict()
        
        # Convert to DataFrame
        features_df = pd.DataFrame(features_dict)
        
        # Prepare features for prediction (same transformations as training)
        features_df = prepare_features_for_training(features_df)
        features_df = features_df.fillna(0)
        
        return features_df
    
    def predict(self, customer_ids):
        """Predict churn probability for customers"""
        
        # Get features
        features_df = self.get_features_for_customers(customer_ids)
        
        # Scale features
        features_scaled = self.scaler.transform(features_df)
        
        # Make predictions
        churn_probabilities = self.model.predict_proba(features_scaled)[:, 1]
        churn_predictions = self.model.predict(features_scaled)
        
        # Create results dataframe
        results = pd.DataFrame({
            'customer_id': customer_ids,
            'churn_probability': churn_probabilities,
            'churn_prediction': churn_predictions,
            'risk_level': pd.cut(churn_probabilities, 
                                bins=[0, 0.3, 0.7, 1.0],
                                labels=['Low', 'Medium', 'High'])
        })
        
        return results
    
    def explain_prediction(self, customer_id):
        """Explain feature contributions for a single customer"""
        
        features_df = self.get_features_for_customers([customer_id])
        features_scaled = self.scaler.transform(features_df)
        
        # Get prediction and feature contributions
        probability = self.model.predict_proba(features_scaled)[0, 1]
        
        # Get feature importance for this prediction
        feature_importance = pd.DataFrame({
            'feature': features_df.columns,
            'value': features_df.iloc[0].values,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\n📊 Prediction Explanation for Customer {customer_id}")
        print(f"  Churn Probability: {probability:.2%}")
        print(f"  Risk Level: {'High' if probability > 0.7 else 'Medium' if probability > 0.3 else 'Low'}")
        
        print(f"\n  Top 5 Contributing Factors:")
        for _, row in feature_importance.head(5).iterrows():
            print(f"    • {row['feature']}: {row['value']:.2f} (importance: {row['importance']:.3f})")
        
        return probability

def run_prediction_demo():
    """Demo of making predictions using the feature store"""
    
    print("\n" + "="*60)
    print("🎯 CHURN PREDICTION DEMO")
    print("="*60)
    
    # Initialize predictor
    predictor = ChurnPredictor()
    
    # Test with some customer IDs
    test_customers = [1, 25, 50, 75, 100]
    
    print(f"\n🔮 Predicting churn for customers: {test_customers}")
    results = predictor.predict(test_customers)
    
    print("\n📊 Prediction Results:")
    print(results.to_string(index=False))
    
    # Show high-risk customers
    high_risk = results[results['risk_level'] == 'High']
    if len(high_risk) > 0:
        print(f"\n⚠️  High Risk Customers (need immediate attention):")
        print(high_risk[['customer_id', 'churn_probability']].to_string(index=False))
    
    # Explain prediction for first customer
    predictor.explain_prediction(test_customers[0])
    
    # Summary statistics
    print(f"\n📈 Prediction Summary:")
    print(f"  Average churn probability: {results['churn_probability'].mean():.2%}")
    print(f"  Customers at high risk: {len(high_risk)}")
    print(f"  Customers at medium risk: {len(results[results['risk_level'] == 'Medium'])}")
    print(f"  Customers at low risk: {len(results[results['risk_level'] == 'Low'])}")
    
    return results

if __name__ == "__main__":
    results = run_prediction_demo()
