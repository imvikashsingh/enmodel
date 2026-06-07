from feast import Entity, Feature, FeatureView, ValueType, FileSource
from feast.types import Float32, Int64, String
from datetime import timedelta

# Define entities
customer = Entity(
    name="customer_id",
    value_type=ValueType.INT64,
    description="Unique customer identifier",
)

# Define data sources
customer_source = FileSource(
    path="data/raw_customer_features.csv",
    event_timestamp_column="event_timestamp",
)

transaction_source = FileSource(
    path="data/raw_transaction_features.csv",
    event_timestamp_column="event_timestamp",
)

# Feature View 1: Customer demographic features
customer_demographic_fv = FeatureView(
    name="customer_demographic_features",
    entities=["customer_id"],
    ttl=timedelta(days=365),
    features=[
        Feature(name="age", dtype=Int64),
        Feature(name="income", dtype=Int64),
        Feature(name="credit_score", dtype=Int64),
        Feature(name="account_age_days", dtype=Int64),
    ],
    batch_source=customer_source,
    online=True,
    tags={"category": "demographic", "importance": "high"},
)

# Feature View 2: Customer banking behavior
customer_behavior_fv = FeatureView(
    name="customer_behavior_features",
    entities=["customer_id"],
    ttl=timedelta(days=365),
    features=[
        Feature(name="num_products", dtype=Int64),
        Feature(name="has_credit_card", dtype=Int64),
        Feature(name="is_active_member", dtype=Int64),
        Feature(name="num_transactions_last_30d", dtype=Int64),
        Feature(name="avg_monthly_balance", dtype=Float32),
    ],
    batch_source=customer_source,
    online=True,
    tags={"category": "behavioral", "importance": "high"},
)

# Feature View 3: Transaction aggregated features (using DuckDB for aggregation)
# Note: For complex aggregations, you'd typically use the offline store
transaction_summary_fv = FeatureView(
    name="transaction_summary_features",
    entities=["customer_id"],
    ttl=timedelta(days=90),
    features=[
        Feature(name="avg_transaction_amount", dtype=Float32),
        Feature(name="max_transaction_amount", dtype=Float32),
        Feature(name="total_transaction_amount", dtype=Float32),
        Feature(name="transaction_count", dtype=Int64),
        Feature(name="online_transaction_ratio", dtype=Float32),
        Feature(name="weekend_transaction_ratio", dtype=Float32),
    ],
    batch_source=transaction_source,
    online=True,
    tags={"category": "transactional", "importance": "medium"},
)

# Feature View 4: Churn label (target variable)
churn_label_fv = FeatureView(
    name="churn_label",
    entities=["customer_id"],
    ttl=timedelta(days=365),
    features=[
        Feature(name="churned", dtype=Int64),
    ],
    batch_source=customer_source,
    online=True,
    tags={"category": "target", "importance": "high"},
)

# Feature View 5: Derived risk features
customer_risk_fv = FeatureView(
    name="customer_risk_features",
    entities=["customer_id"],
    ttl=timedelta(days=365),
    features=[
        Feature(name="credit_score_category", dtype=String),
        Feature(name="account_age_category", dtype=String),
        Feature(name="is_high_risk", dtype=Int64),
    ],
    batch_source=customer_source,
    online=True,
    tags={"category": "risk", "importance": "medium"},
)
